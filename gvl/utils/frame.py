from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from PIL import ImageDraw, ImageFont

from gvl.results.prediction import PredictionRecord
from gvl.utils.aliases import ImageNumpy, ImageT
from gvl.utils.data_types import Episode
from gvl.utils.images import to_pil


def get_frame_from_episode(
    episode: Episode,
    *,
    original_index: int | None = None,
    shuffled_index: int | None = None,
) -> ImageNumpy:
    """Return a frame from an episode by original or shuffled index."""
    if (original_index is None) == (shuffled_index is None):
        raise ValueError("Provide exactly one of original_index or shuffled_index.")
    if shuffled_index is not None:
        return episode.shuffled_frames[shuffled_index]
    index_to_frame = dict(zip(episode.shuffled_frames_indices, episode.shuffled_frames, strict=False))
    if original_index not in index_to_frame:
        raise ValueError(f"Frame index {original_index} not found in episode.")
    return index_to_frame[original_index]


def _format_progress_label(prefix: str, value: int | None) -> str:
    if value is None:
        return f"{prefix}: N/A"
    return f"{prefix}: {value}%"


def _annotate_frame(image: ImageT, label: str):
    pil_image = to_pil(image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    pad = 4
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)
    draw.rectangle((0, 0, text_w + pad * 2, text_h + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return pil_image


def _save_episode_frames(
    frames: Sequence[ImageT],
    progress_values: Sequence[int] | None,
    output_dir: Path,
    label_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = list(progress_values) if progress_values is not None else []
    if len(values) != len(frames):
        logger.warning(
            f"Progress count mismatch for {output_dir.name}: frames={len(frames)} progress_values={len(values)}"
        )
    for idx, frame in enumerate(frames):
        value = values[idx] if idx < len(values) else None
        label = _format_progress_label(label_prefix, value)
        annotated = _annotate_frame(frame, label)
        annotated.save(output_dir / f"frame_{idx:03d}.png")


def save_frame_visualizations(records: list[PredictionRecord], output_root: Path) -> None:
    if not records:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        case_dir = output_root / f"case_{record.index:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for ctx_idx, ep in enumerate(record.example.context_episodes):
            ctx_dir = case_dir / f"context_{ctx_idx:02d}_episode_{ep.episode_index}"
            _save_episode_frames(ep.shuffled_frames, ep.shuffled_frames_approx_completion_rates, ctx_dir, "progress")
        eval_ep = record.example.eval_episode
        gt_dir = case_dir / f"eval_episode_{eval_ep.episode_index}_gt"
        _save_episode_frames(eval_ep.shuffled_frames, eval_ep.shuffled_frames_approx_completion_rates, gt_dir, "gt")
        pred_dir = case_dir / f"eval_episode_{eval_ep.episode_index}_pred"
        _save_episode_frames(eval_ep.shuffled_frames, eval_ep.shuffled_frames_predicted_completion_rates, pred_dir, "pred")
