from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from gvl.utils.aliases import ImageNumpy
from gvl.utils.errors import OriginalFramesLengthMismatch, ShuffledFramesIndicesNotSubset, ShuffledFramesLengthMismatch


def _normalize_anchor_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value == "first":
        return "start"
    if value in {"start", "middle", "last"}:
        return value
    raise ValueError(f"Unknown anchor kind: {kind}")


def _default_anchor_kinds(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["start"]
    if count == 2:
        return ["start", "last"]
    return ["start"] + ["middle"] * (count - 2) + ["last"]


@dataclass
class Episode:
    """
    Container for a single episode (or a selected subsequence of it) used in
    evaluation/in context learning.

    Attributes
    - instruction: Natural-language description of the task to complete.
    - anchor_frames: Anchor frames for prompt context (zero or more).
    - anchor_kinds: Anchor kinds aligned with anchor_frames (start, middle, last).
    - episode_index: Index of this episode within the source dataset.
    - original_frames_indices: Sorted indices from the original episode that
        define the selected subsequence.
    - original_frames_task_completion_rates: Per-frame task completion rates for
        the frames referenced by ``original_frames_indices`` (1:1 aligned; i-th
        value corresponds to the i-th index above).
    - shuffled_frames_indices: Indices from the original episode corresponding to
        ``shuffled_frames``, ordered as they are fed to the model (shuffled order).
        Each entry should also exist in ``original_frames_indices``.
    - shuffled_frames: Frames arranged according to ``shuffled_frames_indices``.
    - shuffled_frames_approx_completion_rates: Per-shuffled-frame approximate
        completion rates (1:1 aligned with ``shuffled_frames``).

    Invariants
    - len(original_frames_indices) == len(original_frames_task_completion_rates)
    - len(shuffled_frames_indices) == len(shuffled_frames)
        == len(shuffled_frames_approx_completion_rates)
    - All values in ``shuffled_frames_indices`` refer to frames from the same
        episode namespace as ``original_frames_indices``.
    """

    instruction: str
    anchor_frames: list[ImageNumpy]
    episode_index: int
    original_frames_indices: list[int]  # subsequence of original episode indices, sorted
    shuffled_frames_indices: list[int]  # original-episode indices in model input (shuffled) order
    shuffled_frames_approx_completion_rates: list[int]  # aligned 1:1 with shuffled_frames
    original_frames_task_completion_rates: list[int]  # aligned 1:1 with original_frames_indices
    shuffled_frames: list[ImageNumpy]  # frames ordered per shuffled_frames_indices
    anchor_kinds: list[str] | None

    def __post_init__(self):
        if self.anchor_frames is None:
            raise ValueError("anchor_frames cannot be None")
        if isinstance(self.anchor_frames, (list, tuple)):
            self.anchor_frames = list(self.anchor_frames)
        else:
            self.anchor_frames = [self.anchor_frames]
        if self.anchor_kinds is None:
            self.anchor_kinds = _default_anchor_kinds(len(self.anchor_frames))
        elif isinstance(self.anchor_kinds, (list, tuple)):
            self.anchor_kinds = [_normalize_anchor_kind(k) for k in self.anchor_kinds]
        else:
            self.anchor_kinds = [_normalize_anchor_kind(self.anchor_kinds)]
        if len(self.anchor_kinds) != len(self.anchor_frames):
            raise ValueError("anchor_kinds length must match anchor_frames length")
        if len(self.original_frames_indices) != len(self.original_frames_task_completion_rates):
            raise OriginalFramesLengthMismatch(len(self.original_frames_indices), len(self.original_frames_task_completion_rates))
        if not (len(self.shuffled_frames_indices) == len(self.shuffled_frames) == len(self.shuffled_frames_approx_completion_rates)):
            raise ShuffledFramesLengthMismatch(
                len(self.shuffled_frames_indices),
                len(self.shuffled_frames),
                len(self.shuffled_frames_approx_completion_rates),
            )
        # Optional: ensure shuffled indices are a subset of original indices
        if not set(self.shuffled_frames_indices).issubset(set(self.original_frames_indices)):
            raise ShuffledFramesIndicesNotSubset()


@dataclass
class InferredEpisode(Episode):
    """
    Extension of Episode that includes model-predicted completion rates for
    the shuffled frames.
    """

    shuffled_frames_predicted_completion_rates: list[int]  # should be aligned 1:1 with shuffled_frames
    # if not, that means that model failed to predict for all frames (e.g. returned incomplete list of preds)

    @classmethod
    def from_predictions(cls, episode: Episode, predictions: list[int]) -> "InferredEpisode":
        """Simple factory method to create an InferredEpisode from an Episode and predictions."""
        return cls(
            instruction=episode.instruction,
            anchor_frames=episode.anchor_frames,
            episode_index=episode.episode_index,
            original_frames_indices=episode.original_frames_indices,
            shuffled_frames_indices=episode.shuffled_frames_indices,
            shuffled_frames_approx_completion_rates=episode.shuffled_frames_approx_completion_rates,
            original_frames_task_completion_rates=episode.original_frames_task_completion_rates,
            shuffled_frames=episode.shuffled_frames,
            shuffled_frames_predicted_completion_rates=predictions,
            anchor_kinds=episode.anchor_kinds,
        )


@dataclass
class ContextEpisodes(Sequence[Episode]):
    """Container for context episodes used as in-context examples."""

    episodes: list[Episode]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int | slice) -> Episode | list[Episode]:
        return self.episodes[index]


@dataclass
class EvalFrame:
    """
    Container for a single evaluation frame.

    Attributes
    - instruction: Natural-language description of the task to complete.
    - frame: The evaluation frame to label or predict.
    - anchor_frames: Optional anchor frame(s) for prompt context.
    - anchor_kinds: Optional anchor kinds aligned with anchor_frames.
    - task_completion_rate: Optional ground-truth completion rate (if known).
    """

    instruction: str
    frame: ImageNumpy
    anchor_frames: list[ImageNumpy] | None = None
    anchor_kinds: list[str] | None = None

    def __post_init__(self) -> None:
        if self.anchor_frames is None:
            self.anchor_kinds = None
            return
        if isinstance(self.anchor_frames, (list, tuple)):
            self.anchor_frames = list(self.anchor_frames)
        else:
            self.anchor_frames = [self.anchor_frames]
        if self.anchor_kinds is None:
            self.anchor_kinds = _default_anchor_kinds(len(self.anchor_frames))
        elif isinstance(self.anchor_kinds, (list, tuple)):
            self.anchor_kinds = [_normalize_anchor_kind(k) for k in self.anchor_kinds]
        else:
            self.anchor_kinds = [_normalize_anchor_kind(self.anchor_kinds)]
        if len(self.anchor_kinds) != len(self.anchor_frames):
            raise ValueError("anchor_kinds length must match anchor_frames length")
    task_completion_rate: int | None = None


@dataclass
class InferredFrame(EvalFrame):
    """EvalFrame with a model-predicted completion rate."""

    predicted_task_completion_rate: int | None = None

    @classmethod
    def from_prediction(cls, frame: EvalFrame, prediction: int | None) -> "InferredFrame":
        return cls(
            instruction=frame.instruction,
            frame=frame.frame,
            anchor_frames=frame.anchor_frames,
            task_completion_rate=frame.task_completion_rate,
            predicted_task_completion_rate=prediction,
            anchor_kinds=frame.anchor_kinds,
        )


@dataclass(kw_only=True)
class EvalCase:
    """Base class for eval cases with context episodes."""

    context_episodes: ContextEpisodes

    def __post_init__(self) -> None:
        if not isinstance(self.context_episodes, ContextEpisodes):
            self.context_episodes = ContextEpisodes(list(self.context_episodes))

    def _context_summary(self) -> tuple[int, list[int], int]:
        ctx_count = len(self.context_episodes)
        ctx_frames_list = [len(ep.shuffled_frames) for ep in self.context_episodes]
        ctx_frames_total = sum(ctx_frames_list)
        return ctx_count, ctx_frames_list, ctx_frames_total


@dataclass
class EpisodeEvalCase(EvalCase):
    """
    Container for a single training/eval case consisting of one
    evaluation episode and multiple context episodes.
    """

    eval_episode: Episode

    def __post_init__(self) -> None:
        super().__post_init__()

    def __repr__(self) -> str:
        eval_frames = len(self.eval_episode.shuffled_frames)
        ctx_count, ctx_frames_list, ctx_frames_total = self._context_summary()
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"eval_episode_index={self.eval_episode.episode_index}, "
            f"eval_frames={eval_frames}, "
            f"context_episodes={ctx_count}, "
            f"context_frames_per_episode={ctx_frames_list}, "
            f"context_frames_total={ctx_frames_total}"
            ")"
        )


@dataclass
class FrameEvalCase(EvalCase):
    """
    Container for a single evaluation frame and optional context episodes.
    """

    eval_frame: EvalFrame

    def __post_init__(self) -> None:
        super().__post_init__()

    def __repr__(self) -> str:
        ctx_count, ctx_frames_list, ctx_frames_total = self._context_summary()
        instruction = self.eval_frame.instruction.replace("\n", " ")
        if len(instruction) > 32:
            instruction = f"{instruction[:29]}..."
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"instruction={instruction!r}, "
            f"context_episodes={ctx_count}, "
            f"context_frames_per_episode={ctx_frames_list}, "
            f"context_frames_total={ctx_frames_total}"
            ")"
        )


@dataclass
class InferredEpisodeEvalCase(EpisodeEvalCase):
    """
    Episode eval case with model-predicted completion rates.
    """

    eval_episode: InferredEpisode

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class InferredFrameEvalCase(FrameEvalCase):
    """Frame eval case with a model-predicted completion rate."""

    eval_frame: InferredFrame

    def __post_init__(self) -> None:
        super().__post_init__()
