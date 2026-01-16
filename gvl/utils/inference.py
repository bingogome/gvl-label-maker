import json
import math
import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from gvl.clients.base import BaseModelClient
from gvl.metrics.base import MetricResult
from gvl.metrics.voc import VOCMetric
from gvl.results.prediction import PredictionRecord
from gvl.utils.constants import N_DEBUG_PROMPT_CHARS
from gvl.utils.data_types import EpisodeEvalCase as FewShotInput
from gvl.utils.data_types import InferredEpisode, InferredEpisodeFewShotResult
from gvl.utils.errors import PercentagesCountMismatch, PercentagesNormalizationError
from gvl.utils.hydra import ensure_required_keys
from gvl.utils.prompts import format_prompt
from gvl.mapper.base import BaseMapper


def build_inferred_case(
    case: FewShotInput,
    predicted: list[int],
) -> InferredEpisodeFewShotResult:
    inferred_ep = InferredEpisode.from_predictions(case.eval_episode, predictions=predicted)
    return InferredEpisodeFewShotResult(eval_episode=inferred_ep, context_episodes=case.context_episodes)


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


def load_fewshot_cases(loader, n: int, dataset_name: str) -> list[FewShotInput]:
    """Load N few-shot cases from a data loader with logging.

    Args:
        loader: Instance of BaseDataLoader.
        n: Number of cases to load.
        dataset_name: Human-friendly dataset identifier for logs.
    Returns:
        List of FewShotInput objects.
    """
    logger.info(f"Generating {n} cases…")
    cases: list[FewShotInput] = []
    for i in range(n):
        logger.info(f"Loading case {i + 1}/{n}")
        case = loader.load_fewshot_input()
        cases.append(case)
    logger.success(f"Loaded {len(cases)} few-shot cases from dataset '{dataset_name}'")
    return cases


def predict_on_fewshot_case(
    idx: int,
    total: int,
    case: FewShotInput,
    client: BaseModelClient,
    prompt_template: str,
    save_raw: bool,
    voc_metric: VOCMetric,
    dataset_name: str,
    temperature: float,
    mapper: BaseMapper,
    *,
    prompt_phrases: dict[str, str] | None = None,
) -> PredictionRecord:
    """Run model prediction and metric computation on a single few-shot case.

    The logic mirrors the original script function without changes.
    """
    logger.info(f"Processing case {idx + 1}/{total} (episode_index={case.eval_episode.episode_index}) from {dataset_name}")
    prompt = format_prompt(prompt_template, instruction=case.eval_episode.instruction)
    logger.debug(f"Prompt (truncated to {N_DEBUG_PROMPT_CHARS} chars): {prompt[:N_DEBUG_PROMPT_CHARS]}...")
    try:
        response_text = client.generate_response(
            prompt,
            case.eval_episode,
            case.context_episodes,
            temperature=temperature,
            prompt_phrases=(prompt_phrases or {}),
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Model generation failed for case {idx}: {e}")
        predicted: list[int] = []
        response_text = f"<error: {e}>"
    logger.debug(f"Response on case {idx}:\n{response_text}")

    expected_len = len(case.eval_episode.shuffled_frames)
    error_count: dict[str, int] = {
        PercentagesCountMismatch.__name__: 0,
        PercentagesNormalizationError.__name__: 0,
    }

    try:
        predicted = mapper.extract_percentages(response_text)
        logger.success(f"Extracted {len(predicted)} percentages on case {idx}")
    except PercentagesNormalizationError as e:
        logger.error(f"Extraction error on case {idx}: {e}")
        predicted = []
        error_count[PercentagesNormalizationError.__name__] += 1

    if len(predicted) != expected_len:
        logger.error(
            f"Count mismatch on case {idx}: expected {expected_len}, "
            f"got {len(predicted)}"
        )
        error_count[PercentagesCountMismatch.__name__] += 1

    inferred: InferredEpisodeFewShotResult = build_inferred_case(case, predicted)

    if sum(error_count.values()) > 0:
        metric_res = MetricResult(name=voc_metric.name, value=0, details={
            "note": f"errors in prediction prevented metric computation {error_count!s}"
        })
    else:
        metric_res = voc_metric.compute(inferred)
    metrics_payload = {metric_res.name: metric_res.value}

    if metric_res.details:
        for k, v in metric_res.details.items():
            metrics_payload[f"{metric_res.name}_{k}"] = v

    logger.debug(
        f"Metrics case {idx}: {metric_res.name}="
        f"{(metric_res.value if metric_res.value is not None else float('nan')):.4f}"
        f"{(' details=' + str(metric_res.details)) if metric_res.details else ''}"
    )

    record = PredictionRecord(
        index=idx,
        dataset=dataset_name,
        example=inferred,
        predicted_percentages=predicted,
        valid_length=len(predicted) == len(case.eval_episode.shuffled_frames),
        metrics=metrics_payload,
        raw_response=response_text if save_raw else None,
        error_count=error_count,
    )
    logger.info(f"Case {idx}: preds={len(predicted)}/{len(case.eval_episode.shuffled_frames)} VOC={metric_res.value}")
    return record
