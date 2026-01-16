"""Benchmarking script producing model inferences + metrics.

Steps:
1. Instantiate data loader & model client via Hydra.
2. Sample N eval cases (EpisodeEvalCase) from loader.
3. For each eval case, call the shared prediction helper.
4. Persist JSONL outputs (one line per eval case) + aggregated metrics summary.
"""

import json
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from datetime import datetime

from gvl.clients.base import BaseModelClient
from gvl.data_loaders.base import BaseDataLoader
from gvl.metrics.voc import VOCMetric
from gvl.results.prediction import aggregate_metrics
from gvl.utils import inference as infer_utils
from gvl.utils.frame import order_episode_frames, save_frame_visualizations, save_progress_video
from gvl.mapper.base import BaseMapper


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict")
def main(config: DictConfig) -> None:
    """Main prediction script entry point."""
    infer_utils.validate_prediction_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    mapper: BaseMapper = instantiate(config.mapper)
    prompt_template: str = config.prompts.template
    
    logger.info(
        f"Instantiated components | dataset={config.dataset.name} loader={data_loader.__class__.__name__} "
        f"model={client.__class__.__name__} prompt_template_chars={len(prompt_template)}"
    )

    num_eval_cases = int(config.prediction.num_eval_cases)
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().isoformat().replace(':', '-')
    jsonl_path = output_dir / f"{model_name_safe}_{starting_time}_predictions.jsonl"
    sampling_method = config.sampling_method
    anchoring = config.anchoring

    eval_cases = infer_utils.load_episode_eval_cases(data_loader, num_eval_cases, config.dataset.name)
    logger.info(
        f"Loaded {len(eval_cases)} (in-context trajectories (0 or more) + eval trajectory) eval cases for prediction"
    )
    if len(eval_cases) == 0:
        logger.warning("No eval cases loaded; exiting")
        return
    voc_metric = VOCMetric()
    logger.debug(f"Metrics initialized: {voc_metric.name}")

    # Load prompt phrasing from dedicated config section (required; fall back to empty)
    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    logger.debug(f"Prompt phrases: {prompt_phrases}")
    records = [
        infer_utils.predict_on_episode_eval_case(
            idx,
            num_eval_cases,
            eval_case,
            client,
            prompt_template,
            save_raw,
            voc_metric,
            config.dataset.name,
            temperature=float(config.prediction.get("temperature", 1.0)),
            mapper=mapper,
            prompt_phrases=prompt_phrases,
        )
        for idx, eval_case in tqdm(enumerate(eval_cases), total=num_eval_cases, desc="Predicting")
    ]

    save_images = bool(config.prediction.get("save_images", True))
    save_videos = bool(config.prediction.get("save_videos", True))
    frames_dir = None
    if save_images or save_videos:
        frames_dir = output_dir / f"{model_name_safe}_{starting_time}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    if save_images and frames_dir is not None:
        logger.info(f"Saving labeled frames to {frames_dir}")
        save_frame_visualizations(records, frames_dir)
    if save_videos and frames_dir is not None:
        video_fps = int(config.prediction.get("video_fps", 2))
        for record in records:
            eval_ep = record.example.eval_episode
            eval_case_dir = frames_dir / f"eval_case_{record.index:04d}"
            eval_case_dir.mkdir(parents=True, exist_ok=True)
            video_path = eval_case_dir / f"eval_episode_{eval_ep.episode_index}_gt_original.mp4"
            video_frames, video_values = order_episode_frames(
                eval_ep,
                eval_ep.shuffled_frames_approx_completion_rates,
                order="original",
            )
            logger.info(f"Saving ground-truth video in 'original' order at {video_path}")
            try:
                save_progress_video(
                    video_frames,
                    video_values,
                    video_path,
                    label_prefix="gt",
                    fps=video_fps,
                )
            except Exception as exc:
                logger.exception(f"Failed to save ground-truth video at {video_path}: {exc}")
                logger.error("MP4 saving requires imageio + imageio-ffmpeg (and ffmpeg).")
            else:
                logger.info(f"Wrote ground-truth video to {video_path}")

    logger.info(f"Serializing {len(records)} prediction records to {jsonl_path}")
    jsonl_payload_iter = (r.to_dict(include_images=False) for r in records)
    infer_utils.save_jsonl(jsonl_payload_iter, jsonl_path)
    dataset_metrics = aggregate_metrics(records)
    logger.success(
        f"Aggregate metrics: total={dataset_metrics.total_examples} valid={dataset_metrics.valid_predictions} "
        f"ratio={(dataset_metrics.length_valid_ratio if dataset_metrics.length_valid_ratio is not None else 0.0):.2f} "
        f"voc_mean={dataset_metrics.metric_means.get('voc', float('nan')):.4f}"
    )
    summary = dict()
    summary['model_name'] = client.model_name
    summary['dataset_name'] = config.dataset.name
    summary['num_context_episodes'] = config.dataset.num_context_episodes
    summary['prediction_time'] = starting_time
    summary['temperature'] = float(config.prediction.get("temperature", 1.0))
    summary['num_eval_cases'] = len(records)
    summary['sampling'] = sampling_method
    summary['metrics'] = dataset_metrics.to_dict()
    summary['prompt_type'] = config.prompts.name
    summary['anchoring'] = anchoring

    with (output_dir / f"{model_name_safe}_{starting_time}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {len(records)} records to {jsonl_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
