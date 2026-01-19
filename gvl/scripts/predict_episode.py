"""Single-episode prediction script (EpisodeEvalCase)."""

import json
import os
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
from gvl.metrics.voc import VOCMetric
from gvl.utils import inference as infer_utils
from gvl.utils.frame import order_episode_frames, save_progress_video


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict_episode")
def main(config: DictConfig) -> None:
    infer_utils.validate_prediction_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    prompt_log_dir = config.get("prompt_log_dir", None)
    if prompt_log_dir:
        os.environ["GVL_CONVERSATION_LOG_DIR"] = str(prompt_log_dir)

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    mapper: BaseMapper = instantiate(config.mapper)
    prompt_template: str = config.prompts.template

    episode_index = int(config.prediction.episode_index)
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().isoformat().replace(":", "-")
    dataset_name = config.dataset.name

    eval_case = data_loader.load_fewshot_input(episode_index=episode_index)
    logger.info(f"Loaded eval case for episode_index={episode_index} from {dataset_name}")

    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    voc_metric = VOCMetric()
    record = infer_utils.predict_on_episode_eval_case(
        idx=0,
        total=1,
        eval_case=eval_case,
        client=client,
        prompt_template=prompt_template,
        save_raw=save_raw,
        voc_metric=voc_metric,
        dataset_name=dataset_name,
        temperature=float(config.prediction.get("temperature", 1.0)),
        mapper=mapper,
        prompt_phrases=prompt_phrases,
    )

    json_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_prediction.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record.to_dict(include_images=False), f, indent=2)
    logger.info(f"Wrote prediction payload to {json_path}")

    video_fps = int(config.prediction.get("video_fps", 2))
    video_order = str(config.prediction.get("video_order", "shuffled")).lower()
    video_path = output_dir / f"{model_name_safe}_{starting_time}_episode_{episode_index}_pred.mp4"
    video_frames, video_values = order_episode_frames(
        record.example.eval_episode,
        record.example.eval_episode.shuffled_frames_predicted_completion_rates,
        order=video_order,
    )
    logger.info(f"Saving prediction video in '{video_order}' order at {video_path}")
    try:
        save_progress_video(
            video_frames,
            video_values,
            video_path,
            label_prefix="pred",
            fps=video_fps,
        )
    except Exception as exc:
        logger.exception(f"Failed to save prediction video at {video_path}: {exc}")
        logger.error("MP4 saving requires imageio + imageio-ffmpeg (and ffmpeg).")
    else:
        logger.info(f"Wrote prediction video to {video_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
