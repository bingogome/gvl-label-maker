"""Episode prediction by per-frame inference (FrameEvalCase)."""

import json
import os
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import ImageDraw
from tqdm import tqdm

from gvl.clients.base import BaseModelClient
from gvl.data_loaders.base import BaseDataLoader
from gvl.mapper.base import BaseMapper
from gvl.metrics.frame_error import FrameProgressErrorMetric
from gvl.utils import inference as infer_utils
from gvl.metrics.voc import value_order_correlation
from gvl.utils.data_types import ContextEpisodes, EvalFrame, FrameEvalCase
from gvl.utils.frame import save_progress_video
from gvl.utils.images import to_pil
from gvl.utils.cleanup import cleanup_resources


def _normalize_anchoring(anchoring: str | list[str] | None) -> list[str]:
    if anchoring is None:
        return []
    if isinstance(anchoring, str):
        if "," in anchoring:
            return [part.strip() for part in anchoring.split(",") if part.strip()]
        return [anchoring]
    return [str(anchor) for anchor in anchoring]


def _select_anchor_frames(frames, anchoring: str | list[str] | None):
    if not frames:
        return [], []
    anchors: list = []
    anchor_kinds: list[str] = []
    choices = _normalize_anchoring(anchoring)
    if not choices:
        return anchors, anchor_kinds
    seen = set()
    for choice in choices:
        if choice == "first":
            anchor_idx = 0
            anchor_kind = "start"
        elif choice == "last":
            anchor_idx = len(frames) - 1
            anchor_kind = "last"
        elif choice == "middle":
            anchor_idx = len(frames) // 2
            anchor_kind = "middle"
        else:
            raise ValueError(f"Unknown anchoring method: {choice}")
        if anchor_idx in seen:
            continue
        anchors.append(frames[anchor_idx])
        anchor_kinds.append(anchor_kind)
        seen.add(anchor_idx)
    return anchors, anchor_kinds


def _compute_task_completion_rate(frame_index: int, total_frames: int) -> int | None:
    if total_frames <= 0:
        return None
    if total_frames == 1:
        return 100
    return round(frame_index / (total_frames - 1) * 100)


def _aggregate_error_counts(records) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        for key, value in record.error_count.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _save_progress_curve_video(
    frames,
    progress_values,
    output_path: Path,
    *,
    fps: int = 2,
    plot_width: int = 220,
    plot_height: int = 80,
) -> Path:
    """Save a video with a small progress curve overlay on each frame."""
    try:
        import numpy as np
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to save MP4 files; install imageio and imageio-ffmpeg") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames provided for video output.")

    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for idx, frame in enumerate(frames):
            pil = to_pil(frame).convert("RGB")
            draw = ImageDraw.Draw(pil)
            # Plot area in bottom-left
            margin = 10
            left = margin
            bottom = pil.height - margin
            top = bottom - plot_height
            right = left + plot_width
            # Semi-transparent grey background patch for the plot
            patch = ImageDraw.ImageDraw(pil, "RGBA")
            patch.rectangle([left, top, right, bottom], fill=(128, 128, 128, 180))
            draw.rectangle([left, top, right, bottom], outline=(255, 255, 255))
            # Build curve up to current frame using only available predictions
            points = []
            seen = [j for j, v in enumerate(progress_values[: idx + 1]) if v is not None]
            if len(seen) >= 2:
                denom = max(1, len(progress_values) - 1)
                for j in seen:
                    v = progress_values[j]
                    x = left + int(plot_width * j / denom)
                    v_clamped = max(0, min(100, int(v)))  # type: ignore[arg-type]
                    y = bottom - int(plot_height * v_clamped / 100)
                    points.append((x, y))
                if len(points) > 1:
                    draw.line(points, fill=(255, 255, 255), width=2)
            writer.append_data(np.array(pil))
    return output_path


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict_episode_by_frame")
def main(config: DictConfig) -> None:
    infer_utils.validate_prediction_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    prompt_log_dir = config.get("prompt_log_dir", None)
    if prompt_log_dir:
        os.environ["GVL_CONVERSATION_LOG_DIR"] = str(prompt_log_dir)

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    mapper: BaseMapper = instantiate(config.mapper)
    prompt_template: str = config.prompts.template

    episode_index = int(config.prediction.get("episode_index", 0))
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("predict_episode_by_frame uses the full episode; sampling is disabled for eval frames.")

    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().isoformat().replace(":", "-")
    dataset_name = config.dataset.name

    frames, instruction = data_loader.load_episode_frames(episode_index=episode_index)
    if not frames:
        raise ValueError(f"No frames loaded for episode_index={episode_index}")
    logger.info(f"Loaded {len(frames)} frames for episode_index={episode_index} from {dataset_name}")

    anchor_source_index = int(config.prediction.get("anchor_episode_index", 0))
    anchor_frames_source, _ = data_loader.load_episode_frames(episode_index=anchor_source_index)
    if not anchor_frames_source:
        raise ValueError(f"No frames loaded for anchor episode_index={anchor_source_index}")
    anchor_frames, anchor_kinds = _select_anchor_frames(anchor_frames_source, config.anchoring)

    context_indices = config.prediction.get("context_episode_indices", None)
    if context_indices:
        logger.info(f"Using explicit context_episode_indices={context_indices}")
        contexts = []
        original_num_frames = data_loader.num_frames
        try:
            for ctx_idx in context_indices:
                ctx_frames, ctx_instruction = data_loader.load_episode_frames(episode_index=int(ctx_idx))
                if not ctx_frames:
                    logger.warning(f"No frames loaded for context episode_index={ctx_idx}; skipping")
                    continue
                contexts.append(
                    data_loader._build_episode(
                        frames=ctx_frames,
                        instruction=ctx_instruction,
                        episode_index=int(ctx_idx),
                        sampling_method=getattr(data_loader, "sampling_method", "random"),
                        anchoring=getattr(data_loader, "anchoring", "first"),
                    )
                )
        finally:
            data_loader.num_frames = original_num_frames
        context_episodes = ContextEpisodes(contexts)
    else:
        context_episodes = ContextEpisodes([])

    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    frame_metric = FrameProgressErrorMetric()
    frame_records = []
    predicted_values: list[int | None] = [None] * len(frames)

    frame_jsonl_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_frame_predictions.jsonl"
    logger.info(f"Streaming per-frame predictions to {frame_jsonl_path}")

    frame_stride = int(config.prediction.get("frame_stride", 0))
    if frame_stride < 0:
        raise ValueError("prediction.frame_stride must be >= 0")

    total_frames = len(frames)
    frame_step = frame_stride + 1
    selected_positions = list(range(0, total_frames, frame_step))
    logger.info(
        f"Predicting {len(selected_positions)}/{total_frames} frames in original order "
        f"with frame_stride={frame_stride}"
    )

    with frame_jsonl_path.open("w", encoding="utf-8") as f:
        for loop_idx, pos in tqdm(
            enumerate(selected_positions),
            total=len(selected_positions),
            desc="Predicting frames",
        ):
            frame = frames[pos]
            gt_rate = _compute_task_completion_rate(pos, total_frames)
            eval_frame = EvalFrame(
                instruction=instruction,
                frame=frame,
                anchor_frames=anchor_frames,
                anchor_kinds=anchor_kinds,
                task_completion_rate=gt_rate,
            )
            frame_eval_case = FrameEvalCase(eval_frame=eval_frame, context_episodes=context_episodes)
            record = infer_utils.predict_on_frame_eval_case(
                idx=loop_idx,
                total=len(selected_positions),
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
            predicted_values[pos] = record.predicted_percentage
            f.write(json.dumps(record.to_dict(include_images=False), ensure_ascii=False) + "\n")
            f.flush()

    missing_predictions = sum(value is None for value in predicted_values)
    if missing_predictions:
        logger.warning(f"{missing_predictions} frame predictions missing; leaving as None.")

    error_count_total = _aggregate_error_counts(frame_records)
    if missing_predictions:
        error_count_total["MissingPrediction"] = missing_predictions

    # VOC on predicted frames only
    available_idx = [i for i, v in enumerate(predicted_values) if v is not None]
    if sum(error_count_total.values()) > 0 or len(available_idx) < 2:
        metrics_payload = {
            "voc": 0.0,
            "voc_note": "errors in prediction prevented metric computation" if sum(error_count_total.values()) > 0 else "insufficient predictions",
        }
    else:
        preds = [predicted_values[i] for i in available_idx]  # type: ignore[index]
        truth = available_idx
        voc_value = value_order_correlation(preds, truth)
        if voc_value != voc_value:  # NaN
            metrics_payload = {"voc": 0.0, "voc_note": "undefined correlation"}
        else:
            metrics_payload = {"voc": float(voc_value)}

    episode_record = {
        "index": 0,
        "dataset": dataset_name,
        "episode_index": episode_index,
        "predicted_percentages": predicted_values,
        "valid_length": missing_predictions == 0,
        "metrics": metrics_payload,
        "error_count": error_count_total,
    }

    logger.info(f"Wrote per-frame predictions to {frame_jsonl_path}")

    episode_json_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_prediction.json"
    with episode_json_path.open("w", encoding="utf-8") as f:
        json.dump(episode_record, f, indent=2)
    logger.info(f"Wrote episode prediction summary to {episode_json_path}")

    video_fps = int(config.prediction.get("video_fps", 2))
    # Carry forward last known prediction for annotation only
    display_values = []
    last_pred = None
    for v in predicted_values:
        if v is not None:
            last_pred = v
        display_values.append(last_pred)

    video_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_pred.mp4"
    logger.info(f"Saving prediction video in original order at {video_path}")
    try:
        save_progress_video(
            frames,
            display_values,
            video_path,
            label_prefix="pred",
            fps=video_fps,
        )
    except Exception as exc:
        logger.exception(f"Failed to save prediction video at {video_path}: {exc}")
        logger.error("MP4 saving requires imageio + imageio-ffmpeg (and ffmpeg).")
    else:
        logger.info(f"Wrote prediction video to {video_path}")

    curve_video_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_pred_curve.mp4"
    logger.info(f"Saving prediction video with progress curve at {curve_video_path}")
    try:
        _save_progress_curve_video(
            frames,
            predicted_values,
            curve_video_path,
            fps=video_fps,
        )
    except Exception as exc:
        logger.exception(f"Failed to save prediction curve video at {curve_video_path}: {exc}")
        logger.error("MP4 saving requires imageio + imageio-ffmpeg (and ffmpeg).")
    else:
        logger.info(f"Wrote prediction curve video to {curve_video_path}")
    cleanup_resources()


if __name__ == "__main__":  # pragma: no cover
    main()
