"""Single-frame prediction script (FrameEvalCase)."""

import json
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from gvl.clients.base import BaseModelClient
from gvl.data_loaders.base import BaseDataLoader
from gvl.mapper.base import BaseMapper
from gvl.metrics.frame_error import FrameProgressErrorMetric
from gvl.utils import inference as infer_utils
from gvl.utils.data_types import EvalFrame, FrameEvalCase
from gvl.utils.frame import save_frame_with_progress


def _select_starting_frame(frames, anchoring: str):
    if not frames:
        return None
    if anchoring == "first":
        return frames[0]
    if anchoring == "last":
        return frames[-1]
    if anchoring == "middle":
        return frames[len(frames) // 2]
    raise ValueError(f"Unknown anchoring method: {anchoring}")


def _compute_task_completion_rate(frame_index: int, total_frames: int) -> int | None:
    if total_frames <= 0:
        return None
    if total_frames == 1:
        return 100
    return round(frame_index / (total_frames - 1) * 100)


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict_frame")
def main(config: DictConfig) -> None:
    infer_utils.validate_prediction_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    mapper: BaseMapper = instantiate(config.mapper)
    prompt_template: str = config.prompts.template

    episode_index = int(config.prediction.episode_index)
    frame_index = int(config.prediction.frame_index)
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().isoformat().replace(":", "-")
    dataset_name = config.dataset.name

    logger.info(f"Predicting frame_index={frame_index} from episode_index={episode_index}")
    context_episodes = data_loader.load_context_episodes(exclude_index=episode_index)
    logger.info(f"Loaded {len(context_episodes)} context episodes from {dataset_name}")

    raw_frames, instruction = data_loader.load_episode_frames(episode_index=episode_index)
    if frame_index < 0 or frame_index >= len(raw_frames):
        raise IndexError(f"frame_index {frame_index} out of bounds for episode length {len(raw_frames)}")
    eval_frame_img = raw_frames[frame_index]
    starting_frame = _select_starting_frame(raw_frames, str(config.anchoring))
    task_completion_rate = _compute_task_completion_rate(frame_index, len(raw_frames))
    if task_completion_rate is None:
        logger.warning(f"Ground truth completion rate not found for frame_index={frame_index}")

    eval_frame = EvalFrame(
        instruction=instruction,
        frame=eval_frame_img,
        starting_frame=starting_frame,
        task_completion_rate=task_completion_rate,
    )
    frame_eval_case = FrameEvalCase(eval_frame=eval_frame, context_episodes=context_episodes)

    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    frame_metric = FrameProgressErrorMetric()
    record = infer_utils.predict_on_frame_eval_case(
        idx=0,
        total=1,
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

    json_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_frame_{frame_index}_prediction.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record.to_dict(include_images=False), f, indent=2)
    logger.info(f"Wrote prediction payload to {json_path}")

    image_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_frame_{frame_index}_pred.png"
    save_frame_with_progress(
        eval_frame.frame,
        record.predicted_percentage,
        image_path,
        label_prefix="pred",
        ground_truth=eval_frame.task_completion_rate,
    )
    logger.info(f"Wrote prediction image to {image_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
