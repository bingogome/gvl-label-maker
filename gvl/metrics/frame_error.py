from gvl.metrics.base import FrameMetric, MetricResult
from gvl.utils.data_types import InferredFrameEvalCase


class FrameProgressErrorMetric(FrameMetric):
    """Signed difference between predicted and ground-truth progress for a frame."""

    @property
    def name(self) -> str:
        return "frame_error"

    def compute(self, example: InferredFrameEvalCase) -> MetricResult:
        eval_frame = example.eval_frame
        pred = eval_frame.predicted_task_completion_rate
        gt = eval_frame.task_completion_rate
        if pred is None or gt is None:
            return MetricResult(name=self.name, value=0.0, details={"note": "missing prediction or ground truth"})
        return MetricResult(name=self.name, value=float(pred - gt))
