from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from gvl.utils.data_types import InferredEpisodeEvalCase, InferredFrameEvalCase


@dataclass
class MetricResult:
    name: str
    value: float | None
    details: dict[str, Any] | None = None


class EpisodeMetric(ABC):
    """Abstract metric interface for episode eval cases."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, example: InferredEpisodeEvalCase) -> MetricResult:
        pass


class FrameMetric(ABC):
    """Abstract metric interface for frame eval cases."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, example: InferredFrameEvalCase) -> MetricResult:
        pass
