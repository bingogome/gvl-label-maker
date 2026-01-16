"""Episode prediction by per-frame inference (FrameEvalCase)."""

import json
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gvl.clients.base import BaseModelClient
from gvl.data_loaders.base import BaseDataLoader
from gvl.mapper.base import BaseMapper
from gvl.metrics.base import MetricResult
from gvl.metrics.frame_error import FrameProgressErrorMetric
from gvl.metrics.voc import VOCMetric
from gvl.results.prediction import EpisodePredictionRecord
from gvl.utils import inference as infer_utils
from gvl.utils.data_types import EvalFrame, FrameEvalCase
from gvl.utils.frame import order_episode_frames, save_progress_video


def _aggregate_error_counts(records) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        for key, value in record.error_count.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _build_metrics_payload(metric_res: MetricResult) -> dict[str, object]:
    payload: dict[str, object] = {metric_res.name: metric_res.value}
    if metric_res.details:
        for key, value in metric_res.details.items():
            payload[f"{metric_res.name}_{key}"] = value
    return payload


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict_episode_by_frame")
def main(config: DictConfig) -> None:
    infer_utils.validate_prediction_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    mapper: BaseMapper = instantiate(config.mapper)
    prompt_template: str = config.prompts.template

    episode_index = int(config.prediction.episode_index)
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    use_full_episode = bool(config.prediction.get("use_full_episode", False))
    num_frames_override = config.prediction.get("num_frames", None)
    if use_full_episode:
        data_loader.num_frames = 1_000_000
    elif num_frames_override is not None:
        data_loader.num_frames = int(num_frames_override)
    logger.info(
        f"Episode sampling for episode_index={episode_index}: "
        f"use_full_episode={use_full_episode} num_frames_override={num_frames_override}"
    )

    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().isoformat().replace(":", "-")
    dataset_name = config.dataset.name

    eval_case = data_loader.load_fewshot_input(episode_index=episode_index)
    eval_episode = eval_case.eval_episode
    logger.info(f"Loaded eval case for episode_index={episode_index} from {dataset_name}")

    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    frame_metric = FrameProgressErrorMetric()
    frame_records = []
    predicted_values: list[int | None] = []

    total_frames = len(eval_episode.shuffled_frames)
    for idx, (frame, gt_rate) in tqdm(
        enumerate(zip(eval_episode.shuffled_frames, eval_episode.shuffled_frames_approx_completion_rates, strict=False)),
        total=total_frames,
        desc="Predicting frames",
    ):
        eval_frame = EvalFrame(
            instruction=eval_episode.instruction,
            frame=frame,
            starting_frame=eval_episode.starting_frame,
            task_completion_rate=gt_rate,
        )
        frame_eval_case = FrameEvalCase(eval_frame=eval_frame, context_episodes=eval_case.context_episodes)
        record = infer_utils.predict_on_frame_eval_case(
            idx=idx,
            total=total_frames,
            eval_case=frame_eval_case,
            client=client,
            prompt_template=prompt_template,
            save_raw=save_raw,
            frame_metric=frame_metric,
            dataset_name=dataset_name,
            temperature=float(config.prediction.get("temperature", 1.0)),
            mapper=mapper,
            prompt_phrases=prompt_phrases,
        )
        frame_records.append(record)
        predicted_values.append(record.predicted_percentage)

    missing_predictions = sum(value is None for value in predicted_values)
    if missing_predictions:
        logger.warning(f"{missing_predictions} frame predictions missing; filling with 0 for episode summary.")

    predicted_values_int = [value if value is not None else 0 for value in predicted_values]
    inferred_episode = infer_utils.build_inferred_episode_eval_case(eval_case, predicted_values_int)

    error_count_total = _aggregate_error_counts(frame_records)
    if missing_predictions:
        error_count_total["MissingPrediction"] = missing_predictions

    voc_metric = VOCMetric()
    if sum(error_count_total.values()) > 0:
        metric_res = MetricResult(
            name=voc_metric.name,
            value=0.0,
            details={"note": f"errors in prediction prevented metric computation {error_count_total!s}"},
        )
    else:
        metric_res = voc_metric.compute(inferred_episode)
    metrics_payload = _build_metrics_payload(metric_res)

    episode_record = EpisodePredictionRecord(
        index=0,
        dataset=dataset_name,
        example=inferred_episode,
        predicted_percentages=predicted_values_int,
        valid_length=len(predicted_values_int) == len(eval_episode.shuffled_frames),
        metrics=metrics_payload,
        raw_response=None,
        error_count=error_count_total,
    )

    frame_jsonl_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_frame_predictions.jsonl"
    with frame_jsonl_path.open("w", encoding="utf-8") as f:
        for record in frame_records:
            f.write(json.dumps(record.to_dict(include_images=False), ensure_ascii=False) + "\n")
    logger.info(f"Wrote per-frame predictions to {frame_jsonl_path}")

    episode_json_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_prediction.json"
    with episode_json_path.open("w", encoding="utf-8") as f:
        json.dump(episode_record.to_dict(include_images=False), f, indent=2)
    logger.info(f"Wrote episode prediction summary to {episode_json_path}")

    video_fps = int(config.prediction.get("video_fps", 2))
    video_order = str(config.prediction.get("video_order", "shuffled")).lower()
    video_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_pred.mp4"
    video_frames, video_values = order_episode_frames(
        eval_episode,
        predicted_values,
        order=video_order,
    )
    logger.info(f"Saving prediction video in '{video_order}' order at {video_path}")
    try:
        save_progress_video(
            video_frames,
            video_values,
            video_path,
            label_prefix="pred",
            fps=video_fps,
        )
    except Exception as exc:
        logger.exception(f"Failed to save prediction video at {video_path}: {exc}")
        logger.error("MP4 saving requires imageio + imageio-ffmpeg (and ffmpeg).")
    else:
        logger.info(f"Wrote prediction video to {video_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
