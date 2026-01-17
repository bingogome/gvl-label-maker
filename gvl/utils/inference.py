import json
from collections.abc import Callable, Iterable
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from gvl.clients.base import BaseModelClient
from gvl.metrics.base import MetricResult
from gvl.metrics.frame_error import FrameProgressErrorMetric
from gvl.metrics.voc import VOCMetric
from gvl.results.prediction import EpisodePredictionRecord, FramePredictionRecord
from gvl.utils.constants import N_DEBUG_PROMPT_CHARS
from gvl.utils.data_types import EpisodeEvalCase
from gvl.utils.data_types import EvalCase, FrameEvalCase
from gvl.utils.data_types import InferredEpisode, InferredEpisodeEvalCase
from gvl.utils.data_types import InferredFrame, InferredFrameEvalCase
from gvl.utils.errors import PercentagesCountMismatch, PercentagesNormalizationError
from gvl.utils.hydra import ensure_required_keys
from gvl.utils.prompts import format_prompt
from gvl.mapper.base import BaseMapper


def build_inferred_episode_eval_case(
    eval_case: EpisodeEvalCase,
    predicted: list[int],
) -> InferredEpisodeEvalCase:
    inferred_ep = InferredEpisode.from_predictions(eval_case.eval_episode, predictions=predicted)
    return InferredEpisodeEvalCase(eval_episode=inferred_ep, context_episodes=eval_case.context_episodes)


def build_inferred_frame_eval_case(
    eval_case: FrameEvalCase,
    predicted: int | None,
) -> InferredFrameEvalCase:
    inferred_frame = InferredFrame.from_prediction(eval_case.eval_frame, prediction=predicted)
    return InferredFrameEvalCase(eval_frame=inferred_frame, context_episodes=eval_case.context_episodes)


def _extract_percentages(
    response_text: str,
    mapper: BaseMapper,
    expected_len: int,
    *,
    eval_case_label: str,
    idx: int,
) -> tuple[list[int], dict[str, int]]:
    error_count: dict[str, int] = {
        PercentagesCountMismatch.__name__: 0,
        PercentagesNormalizationError.__name__: 0,
    }
    try:
        predicted = mapper.extract_percentages(response_text)
        logger.success(f"Extracted {len(predicted)} percentages on {eval_case_label} {idx}")
    except PercentagesNormalizationError as e:
        logger.error(f"Extraction error on {eval_case_label} {idx}: {e}")
        predicted = []
        error_count[PercentagesNormalizationError.__name__] += 1

    if len(predicted) != expected_len:
        logger.error(
            f"Count mismatch on {eval_case_label} {idx}: expected {expected_len}, "
            f"got {len(predicted)}"
        )
        error_count[PercentagesCountMismatch.__name__] += 1
    return predicted, error_count


def _build_metrics_payload(metric_res: MetricResult) -> dict[str, object]:
    payload: dict[str, object] = {metric_res.name: metric_res.value}
    if metric_res.details:
        for k, v in metric_res.details.items():
            payload[f"{metric_res.name}_{k}"] = v
    return payload


def _log_metric_result(eval_case_label: str, idx: int, metric_res: MetricResult) -> None:
    logger.debug(
        f"Metrics {eval_case_label} {idx}: {metric_res.name}="
        f"{(metric_res.value if metric_res.value is not None else float('nan')):.4f}"
        f"{(' details=' + str(metric_res.details)) if metric_res.details else ''}"
    )


def _generate_eval_case_response(
    *,
    eval_case_label: str,
    idx: int,
    prompt_template: str,
    instruction: str,
    num_frames: int,
    generate_fn: Callable[[str], str],
) -> str:
    prompt = format_prompt(prompt_template, instruction=instruction, num_frames=num_frames)
    logger.debug(f"Prompt (truncated to {N_DEBUG_PROMPT_CHARS} chars): {prompt[:N_DEBUG_PROMPT_CHARS]}...")
    try:
        response_text = generate_fn(prompt)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Model generation failed for {eval_case_label} {idx}: {e}")
        response_text = f"<error: {e}>"
    logger.debug(f"Response on {eval_case_label} {idx}:\n{response_text}")
    return response_text


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_prediction_config(config: DictConfig) -> None:
    """Ensure required top-level keys are present for prediction runs.

    This mirrors the previous local _validate_config in the script.
    """
    for key in ("dataset", "data_loader", "model", "prompts", "prediction"):
        ensure_required_keys(config, key)


def load_episode_eval_cases(loader, n: int, dataset_name: str) -> list[EpisodeEvalCase]:
    """Load N episode eval cases from a data loader with logging.

    Args:
        loader: Instance of BaseDataLoader.
        n: Number of eval cases to load.
        dataset_name: Human-friendly dataset identifier for logs.
    Returns:
        List of EpisodeEvalCase objects.
    """
    logger.info(f"Generating {n} eval cases…")
    eval_cases: list[EpisodeEvalCase] = []
    for i in range(n):
        logger.info(f"Loading eval case {i + 1}/{n}")
        eval_case = loader.load_fewshot_input()
        eval_cases.append(eval_case)
    logger.success(f"Loaded {len(eval_cases)} episode eval cases from dataset '{dataset_name}'")
    return eval_cases


def predict_on_episode_eval_case(
    idx: int,
    total: int,
    eval_case: EpisodeEvalCase,
    client: BaseModelClient,
    prompt_template: str,
    save_raw: bool,
    voc_metric: VOCMetric,
    dataset_name: str,
    temperature: float,
    mapper: BaseMapper,
    *,
    prompt_phrases: dict[str, str | list[str]] | None = None,
) -> EpisodePredictionRecord:
    """Run model prediction and metric computation on a single episode eval case."""
    logger.info(
        f"Processing eval case {idx + 1}/{total} "
        f"(episode_index={eval_case.eval_episode.episode_index}) from {dataset_name}"
    )
    response_text = _generate_eval_case_response(
        eval_case_label="episode eval case",
        idx=idx,
        prompt_template=prompt_template,
        instruction=eval_case.eval_episode.instruction,
        num_frames=len(eval_case.eval_episode.shuffled_frames),
        generate_fn=lambda prompt: client.generate_response_for_episode(
            prompt,
            eval_case.eval_episode,
            eval_case.context_episodes,
            temperature=temperature,
            prompt_phrases=(prompt_phrases or {}),
        ),
    )

    expected_len = len(eval_case.eval_episode.shuffled_frames)
    predicted, error_count = _extract_percentages(
        response_text,
        mapper,
        expected_len,
        eval_case_label="episode eval case",
        idx=idx,
    )

    inferred: InferredEpisodeEvalCase = build_inferred_episode_eval_case(eval_case, predicted)

    if sum(error_count.values()) > 0:
        metric_res = MetricResult(name=voc_metric.name, value=0, details={
            "note": f"errors in prediction prevented metric computation {error_count!s}"
        })
    else:
        metric_res = voc_metric.compute(inferred)
    metrics_payload = _build_metrics_payload(metric_res)
    _log_metric_result("episode eval case", idx, metric_res)

    record = EpisodePredictionRecord(
        index=idx,
        dataset=dataset_name,
        example=inferred,
        predicted_percentages=predicted,
        valid_length=len(predicted) == len(eval_case.eval_episode.shuffled_frames),
        metrics=metrics_payload,
        raw_response=response_text if save_raw else None,
        error_count=error_count,
    )
    logger.info(
        f"Eval case {idx}: preds={len(predicted)}/{len(eval_case.eval_episode.shuffled_frames)} "
        f"VOC={metric_res.value}"
    )
    return record


def predict_on_frame_eval_case(
    idx: int,
    total: int,
    eval_case: FrameEvalCase,
    client: BaseModelClient,
    prompt_template: str,
    save_raw: bool,
    frame_metric: FrameProgressErrorMetric,
    dataset_name: str,
    temperature: float,
    mapper: BaseMapper,
    *,
    prompt_phrases: dict[str, str | list[str]] | None = None,
) -> FramePredictionRecord:
    """Run model prediction and metric computation on a single frame eval case."""
    logger.info(f"Processing frame eval case {idx + 1}/{total} from {dataset_name}")
    response_text = _generate_eval_case_response(
        eval_case_label="frame eval case",
        idx=idx,
        prompt_template=prompt_template,
        instruction=eval_case.eval_frame.instruction,
        num_frames=1,
        generate_fn=lambda prompt: client.generate_response_for_frame(
            prompt,
            eval_case.eval_frame,
            eval_case.context_episodes,
            temperature=temperature,
            prompt_phrases=(prompt_phrases or {}),
        ),
    )

    expected_len = 1
    predicted_list, error_count = _extract_percentages(
        response_text,
        mapper,
        expected_len=expected_len,
        eval_case_label="frame eval case",
        idx=idx,
    )

    predicted_value = predicted_list[0] if predicted_list else None
    if len(predicted_list) > 1:
        logger.warning(f"Frame eval case {idx}: multiple predictions returned, using first value")

    inferred = build_inferred_frame_eval_case(eval_case, predicted_value)
    if sum(error_count.values()) > 0:
        metric_res = MetricResult(name=frame_metric.name, value=0.0, details={
            "note": f"errors in prediction prevented metric computation {error_count!s}"
        })
    else:
        metric_res = frame_metric.compute(inferred)
    metrics_payload = _build_metrics_payload(metric_res)
    _log_metric_result("frame eval case", idx, metric_res)

    record = FramePredictionRecord(
        index=idx,
        dataset=dataset_name,
        example=inferred,
        predicted_percentage=predicted_value,
        valid_value=len(predicted_list) == expected_len,
        metrics=metrics_payload,
        raw_response=response_text if save_raw else None,
        error_count=error_count,
    )
    logger.info(f"Frame eval case {idx}: pred={predicted_value}")
    return record


def predict_on_eval_case(
    idx: int,
    total: int,
    eval_case: EvalCase,
    client: BaseModelClient,
    prompt_template: str,
    save_raw: bool,
    dataset_name: str,
    temperature: float,
    mapper: BaseMapper,
    *,
    frame_metric: FrameProgressErrorMetric = FrameProgressErrorMetric(),
    voc_metric: VOCMetric = VOCMetric(),
    prompt_phrases: dict[str, str | list[str]] | None = None,
) -> EpisodePredictionRecord | FramePredictionRecord:
    """Dispatch prediction based on eval case type."""
    if isinstance(eval_case, EpisodeEvalCase):
        metric = voc_metric
        return predict_on_episode_eval_case(
            idx,
            total,
            eval_case,
            client,
            prompt_template,
            save_raw,
            metric,
            dataset_name,
            temperature,
            mapper,
            prompt_phrases=prompt_phrases,
        )
    if isinstance(eval_case, FrameEvalCase):
        return predict_on_frame_eval_case(
            idx,
            total,
            eval_case,
            client,
            prompt_template,
            save_raw,
            frame_metric,
            dataset_name,
            temperature,
            mapper,
            prompt_phrases=prompt_phrases,
        )
    raise ValueError(f"Unsupported EvalCase type: {type(eval_case)}")
