from collections.abc import Sequence
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from gvl.data_loaders.base import BaseDataLoader
from gvl.utils.data_types import ContextEpisodes, EpisodeEvalCase

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class LocalDataLoader(BaseDataLoader):
    """Load a single episode from local image files.

    By default, treats an entire directory (or an explicit list of files)
    as one episode ordered by filename. The resulting EpisodeEvalCase contains the
    eval episode and no context episodes.
    """

    def __init__(
        self,
        *,
        episodes_files: Sequence[Sequence[str]],
        instruction: str = "",
        num_frames: int = 20,
        num_context_episodes: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        sampling_method: str = 'random',
    ) -> None:
        super().__init__(
            num_frames=num_frames,
            num_context_episodes=num_context_episodes,
            shuffle=shuffle,
            seed=seed,
        )
        if not episodes_files or len(episodes_files) == 0:
            raise ValueError
        # Normalize to absolute Paths at call time to preserve user-specified order
        self.episodes_files: list[list[str]] = [list(ep) for ep in episodes_files]
        self.instruction = instruction or ""
        self.sampling_method = sampling_method

    def _load_images(self, paths: list[Path]):
        images = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning(f"Skipping unreadable image {p}: {exc}")
        return images

    def load_episode_frames(self, episode_index: int) -> tuple[list, str]:
        """Load raw frames and instruction for an episode without sampling or shuffling."""
        if episode_index < 0 or episode_index >= len(self.episodes_files):
            raise IndexError
        paths = [Path(p) for p in self.episodes_files[episode_index]]
        frames = self._load_images(paths)
        if not frames:
            raise ValueError
        return frames, self.instruction

    def load_context_episodes(self, *, exclude_index: int | None = None) -> ContextEpisodes:
        if self.num_context_episodes <= 0:
            return ContextEpisodes([])
        pool = [
            idx for idx in range(len(self.episodes_files))
            if exclude_index is None or idx != exclude_index
        ]
        if not pool:
            return ContextEpisodes([])
        rng_seed = self.seed if exclude_index is None else self.seed + int(exclude_index)
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(pool)
        chosen = pool[: self.num_context_episodes]
        ctx_eps = []
        for idx in chosen:
            paths = [Path(p) for p in self.episodes_files[idx]]
            frames = self._load_images(paths)
            if not frames:
                raise ValueError
            ctx_eps.append(self._build_episode(
                frames=frames,
                instruction=self.instruction,
                episode_index=idx,
                sampling_method=self.sampling_method,
            ))
        return ContextEpisodes(ctx_eps)

    def load_fewshot_input(self, episode_index: int | None = None) -> EpisodeEvalCase:
        if episode_index is None:
            episode_index = 0
        if episode_index < 0 or episode_index >= len(self.episodes_files):
            raise IndexError
        # Do not reorder or auto-discover; respect user-provided order strictly.
        paths = [Path(p) for p in self.episodes_files[episode_index]]
        frames = self._load_images(paths)
        if not frames:
            raise ValueError
        ep = self._build_episode(
            frames=frames,
            instruction=self.instruction,
            episode_index=episode_index or 0,
            sampling_method=self.sampling_method
        )
        return EpisodeEvalCase(eval_episode=ep, context_episodes=ContextEpisodes([]))
