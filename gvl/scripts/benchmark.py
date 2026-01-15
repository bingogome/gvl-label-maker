"""Benchmarking script producing model inferences + metrics.

Steps:
1. Instantiate data loader & model client via Hydra.
2. Sample N cases (FewShotInput) from loader.
3. For each case, call the shared prediction helper.
4. Persist JSONL outputs (one line per case) + aggregated metrics summary.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import ImageDraw, ImageFont
from tqdm import tqdm
from datetime import datetime

from gvl.clients.base import BaseModelClient
from gvl.data_loaders.base import BaseDataLoader
from gvl.metrics.voc import VOCMetric
from gvl.results.prediction import PredictionRecord, aggregate_metrics
from gvl.utils import inference as infer_utils
from gvl.utils.aliases import ImageT
from gvl.utils.images import to_pil
from gvl.mapper.base import BaseMapper


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

    num_cases = int(config.prediction.num_cases)
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().isoformat().replace(':', '-')
    jsonl_path = output_dir / f"{model_name_safe}_{starting_time}_predictions.jsonl"
    sampling_method = config.sampling_method
    anchoring = config.anchoring

    cases = infer_utils.load_fewshot_cases(data_loader, num_cases, config.dataset.name)
    logger.info(f"Loaded {len(cases)} (in-context trajectories (0 or more) + eval trajectory) cases for prediction")
    if len(cases) == 0:
        logger.warning("No cases loaded; exiting")
        return
    voc_metric = VOCMetric()
    logger.debug(f"Metrics initialized: {voc_metric.name}")

    # Load prompt phrasing from dedicated config section (required; fall back to empty)
    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    logger.debug(f"Prompt phrases: {prompt_phrases}")
    records = [
        infer_utils.predict_on_fewshot_case(
            idx,
            num_cases,
            case,
            client,
            prompt_template,
            save_raw,
            voc_metric,
            config.dataset.name,
            temperature=float(config.prediction.get("temperature", 1.0)),
            mapper=mapper,
            prompt_phrases=prompt_phrases,
        )
        for idx, case in tqdm(enumerate(cases), total=num_cases, desc="Predicting")
    ]

    save_images = bool(config.prediction.get("save_images", True))
    if save_images:
        frames_dir = output_dir / f"{model_name_safe}_{starting_time}_frames"
        logger.info(f"Saving labeled frames to {frames_dir}")
        save_frame_visualizations(records, frames_dir)

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
    summary['num_cases'] = len(records)
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
