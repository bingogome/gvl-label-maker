from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from PIL import ImageDraw, ImageFont

from gvl.results.prediction import EpisodePredictionRecord
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


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _annotate_frame(image: ImageT, label: str):
    pil_image = to_pil(image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    pad = 4
    lines = label.splitlines() or [label]
    line_spacing = 2
    max_w = 0
    total_h = 0
    for i, line in enumerate(lines):
        text_w, text_h = _measure_text(draw, line, font)
        max_w = max(max_w, text_w)
        total_h += text_h
        if i < len(lines) - 1:
            total_h += line_spacing
    draw.rectangle((0, 0, max_w + pad * 2, total_h + pad * 2), fill=(0, 0, 0))
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255), font=font)
        _, text_h = _measure_text(draw, line, font)
        y += text_h + line_spacing
    return pil_image


def _save_episode_frames(
    frames: Sequence[ImageT],
    progress_values: Sequence[int | None] | None,
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


def save_frame_with_progress(
    frame: ImageT,
    progress_value: int | None,
    output_path: Path,
    *,
    label_prefix: str = "pred",
    ground_truth: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label = _format_progress_label(label_prefix, progress_value)
    if ground_truth is not None:
        label = f"{label}\n{_format_progress_label('gt', ground_truth)}"
    annotated = _annotate_frame(frame, label)
    annotated.save(output_path)


def save_progress_video(
    frames: Sequence[ImageT],
    progress_values: Sequence[int | None],
    output_path: Path,
    *,
    label_prefix: str = "pred",
    fps: int = 2,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames provided for video output.")
    values = list(progress_values)
    if len(values) != len(frames):
        logger.warning(
            f"Progress count mismatch for video {output_path.name}: frames={len(frames)} progress_values={len(values)}"
        )
    annotated_frames = []
    for idx, frame in enumerate(frames):
        value = values[idx] if idx < len(values) else None
        label = _format_progress_label(label_prefix, value)
        annotated_frames.append(_annotate_frame(frame, label))

    suffix = output_path.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        logger.warning(f"Video output suffix {output_path.suffix} not supported; saving MP4 instead.")
        output_path = output_path.with_suffix(".mp4")
        suffix = ".mp4"

    if suffix == ".gif":
        duration_ms = int(1000 / fps) if fps > 0 else 500
        annotated_frames[0].save(
            output_path,
            save_all=True,
            append_images=annotated_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return output_path

    try:
        import numpy as np
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to save MP4 files; install imageio and imageio-ffmpeg") from exc

    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in annotated_frames:
            writer.append_data(np.array(frame))
    return output_path


def order_episode_frames(
    episode: Episode,
    progress_values: Sequence[int | None],
    *,
    order: str = "shuffled",
) -> tuple[list[ImageT], list[int | None]]:
    frames = list(episode.shuffled_frames)
    values = list(progress_values)
    if order == "shuffled":
        return frames, values
    if order != "original":
        raise ValueError(f"Unsupported frame order: {order}")
    if len(values) != len(frames):
        logger.warning(
            f"Progress count mismatch while ordering by original indices: frames={len(frames)} progress_values={len(values)}"
        )
    order_indices = sorted(
        range(len(episode.shuffled_frames_indices)),
        key=lambda i: episode.shuffled_frames_indices[i],
    )
    ordered_frames = [frames[i] for i in order_indices]
    ordered_values = [values[i] if i < len(values) else None for i in order_indices]
    return ordered_frames, ordered_values


def save_frame_visualizations(records: list[EpisodePredictionRecord], output_root: Path) -> None:
    if not records:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        eval_case_dir = output_root / f"eval_case_{record.index:04d}"
        eval_case_dir.mkdir(parents=True, exist_ok=True)
        for ctx_idx, ep in enumerate(record.example.context_episodes):
            ctx_dir = eval_case_dir / f"context_{ctx_idx:02d}_episode_{ep.episode_index}"
            _save_episode_frames(ep.shuffled_frames, ep.shuffled_frames_approx_completion_rates, ctx_dir, "progress")
        eval_ep = record.example.eval_episode
        gt_dir = eval_case_dir / f"eval_episode_{eval_ep.episode_index}_gt"
        _save_episode_frames(eval_ep.shuffled_frames, eval_ep.shuffled_frames_approx_completion_rates, gt_dir, "gt")
        pred_dir = eval_case_dir / f"eval_episode_{eval_ep.episode_index}_pred"
        _save_episode_frames(eval_ep.shuffled_frames, eval_ep.shuffled_frames_predicted_completion_rates, pred_dir, "pred")
