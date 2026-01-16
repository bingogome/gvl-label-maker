"""Structured prediction result representations and serialization helpers."""

from dataclasses import asdict, dataclass, field
from typing import Any

from gvl.utils.data_types import ContextEpisodes, InferredEpisodeEvalCase, InferredFrameEvalCase


def _context_payload(context_episodes: ContextEpisodes) -> dict[str, Any]:
    ctx_count = len(context_episodes)
    ctx_indices = [ep.episode_index for ep in context_episodes]
    ctx_frames_per_ep = [len(ep.shuffled_frames) for ep in context_episodes]
    return {
        "context_episodes_count": ctx_count,
        "context_episodes_indices": ctx_indices,
        "context_episodes_frames_per_episode": ctx_frames_per_ep,
    }


@dataclass
class BasePredictionRecord:
    """Shared prediction record fields."""

    index: int
    dataset: str
    metrics: dict[str, Any]
    error_count: dict[str, int]
    raw_response: str | None

    def _base_payload(self) -> dict[str, Any]:
        base = {
            "index": self.index,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "error_count": self.error_count,
        }
        if self.raw_response is not None:
            base["raw_response"] = self.raw_response
        return base


@dataclass
class EpisodePredictionRecord(BasePredictionRecord):
    """A single model prediction result for one episode eval case."""

    example: InferredEpisodeEvalCase
    predicted_percentages: list[int]
    valid_length: bool

    def to_dict(self, *, include_images: bool = False) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        Images are omitted by default (cannot JSON serialize numpy arrays)."""

        eval_ep = self.example.eval_episode
        ctx_eps = self.example.context_episodes
        base = self._base_payload()
        base.update({
            "eval_episode": {
                "episode_index": eval_ep.episode_index,
                "instruction": eval_ep.instruction,
                "original_frames_indices": eval_ep.original_frames_indices,
                "shuffled_frames_indices": eval_ep.shuffled_frames_indices,
                "original_frames_task_completion_rates": eval_ep.original_frames_task_completion_rates,
                "shuffled_frames_approx_completion_rates": eval_ep.shuffled_frames_approx_completion_rates,
            },
            "predicted_percentages": self.predicted_percentages,
            "valid_length": self.valid_length,
        })
        base.update(_context_payload(ctx_eps))
        if include_images:
            base["eval_episode"]["_frames_present"] = True
        return base


@dataclass
class FramePredictionRecord(BasePredictionRecord):
    """A single model prediction result for one frame eval case."""

    example: InferredFrameEvalCase
    predicted_percentage: int | None
    valid_value: bool

    def to_dict(self, *, include_images: bool = False) -> dict[str, Any]:
        eval_frame = self.example.eval_frame
        ctx_eps = self.example.context_episodes
        base = self._base_payload()
        base.update({
            "eval_frame": {
                "instruction": eval_frame.instruction,
                "task_completion_rate": eval_frame.task_completion_rate,
                "predicted_task_completion_rate": eval_frame.predicted_task_completion_rate,
            },
            "predicted_percentage": self.predicted_percentage,
            "valid_value": self.valid_value,
        })
        base.update(_context_payload(ctx_eps))
        if include_images:
            base["eval_frame"]["_frame_present"] = True
        return base


@dataclass
class DatasetMetrics:
    total_examples: int
    valid_predictions: int
    length_valid_ratio: float | None
    metric_means: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return asdict(self)


def aggregate_metrics(records: list[EpisodePredictionRecord]) -> DatasetMetrics:
    if not records:
        return DatasetMetrics(total_examples=0, valid_predictions=0, length_valid_ratio=None, metric_means={})
    total = len(records)
    valid = sum(r.valid_length for r in records)
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for r in records:
        for k, v in r.metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                metric_counts[k] = metric_counts.get(k, 0) + 1
    means = {k: metric_sums[k] / metric_counts[k] for k in metric_sums if metric_counts[k] > 0}
    ratio = valid / total if total else None
    return DatasetMetrics(
        total_examples=total,
        valid_predictions=valid,
        length_valid_ratio=ratio,
        metric_means=means,
    )
