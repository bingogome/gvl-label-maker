from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from loguru import logger

from gvl.data_loaders.base import BaseDataLoader
from gvl.utils.data_types import ContextEpisodes, EpisodeEvalCase

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}  # Support multiple video formats


class MP4DataLoader(BaseDataLoader):
    """Load episodes from a folder of MP4 videos plus a shared instruction text file.

    The folder is expected to contain one or more .mp4 files and a single text
    file with the task instruction (shared across all episodes).
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        instruction_file: str = "instruction.txt",
        instruction: str | None = None,
        num_frames: int = 20,
        num_context_episodes: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        sampling_method: str = "random",
        anchoring: str | Sequence[str] | None = "first",
    ) -> None:
        super().__init__(
            num_frames=num_frames,
            num_context_episodes=num_context_episodes,
            shuffle=shuffle,
            seed=seed,
        )
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MP4 data directory not found: {self.data_dir}")

        self.video_paths = self._discover_videos(self.data_dir)
        if not self.video_paths:
            raise ValueError(f"No MP4 files found in {self.data_dir}")
        
        logger.info(f"Discovered {len(self.video_paths)} video(s) in {self.data_dir}")
        for idx, vp in enumerate(self.video_paths):
            logger.debug(f"  [{idx}] {vp.name} ({vp.stat().st_size / 1024:.1f} KB)")

        self.instruction_file = instruction_file
        self._instruction_override = instruction
        self.instruction = self._load_instruction()
        self.sampling_method = sampling_method
        self.anchoring = anchoring

    def _discover_videos(self, data_dir: Path) -> list[Path]:
        paths = [
            p for p in data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
        return sorted(paths)

    def _load_instruction(self) -> str:
        if self._instruction_override is not None:
            return str(self._instruction_override)
        instruction_path = Path(self.instruction_file)
        if not instruction_path.is_absolute():
            instruction_path = self.data_dir / instruction_path
        if not instruction_path.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_path}")
        text = instruction_path.read_text(encoding="utf-8")
        return text.strip()

    def _load_video_frames(self, path: Path) -> list:
        """Load frames from a video file, trying multiple backends for robustness."""
        frames = []
        reader = None
        logger.info(f"Reading video: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
        
        # Try multiple backends in order of preference
        # pyav is most robust, ffmpeg is fallback, None uses imageio's default
        backends = [
            ("pyav", {"format": "rgb24"}),  # PyAV with RGB format
            ("ffmpeg", {}),  # FFmpeg backend
            (None, {}),  # Default backend
        ]
        
        last_error = None
        successful_backend = None
        
        for backend_name, kwargs in backends:
            try:
                if backend_name is None:
                    reader = imageio.get_reader(str(path), **kwargs)
                    successful_backend = "default"
                else:
                    reader = imageio.get_reader(str(path), backend_name, **kwargs)
                    successful_backend = backend_name
                logger.debug(f"Successfully opened {path.name} with {successful_backend} backend")
                break  # Success, exit loop
            except ImportError as e:
                logger.debug(f"Backend {backend_name} not available: {e}")
                last_error = e
                reader = None
                continue
            except Exception as e:
                last_error = e
                logger.debug(f"Backend {backend_name} failed to open {path.name}: {str(e)[:100]}")
                reader = None
                continue
        
        if reader is None:
            error_msg = (
                f"Could not open video {path.name} with any backend.\n"
                f"Last error: {last_error}\n"
                f"Try installing PyAV: pip install av\n"
                f"Or re-encode the video: ffmpeg -i {path.name} -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p output.mp4"
            )
            raise ValueError(error_msg)
        
        try:
            frame_count = 0
            for frame in reader:
                # Ensure 3-channel RGB-like arrays
                if frame.ndim == 2:
                    # Grayscale - convert to RGB
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.ndim == 3:
                    if frame.shape[2] == 1:
                        # Single channel - convert to RGB
                        frame = np.repeat(frame, 3, axis=2)
                    elif frame.shape[2] == 4:
                        # RGBA - drop alpha channel
                        frame = frame[:, :, :3]
                    elif frame.shape[2] > 4:
                        # Unknown format - take first 3 channels
                        frame = frame[:, :, :3]
                    # Ensure contiguous array
                    frame = np.ascontiguousarray(frame)
                frames.append(frame)
                frame_count += 1
            
            logger.info(
                f"Successfully read {frame_count} frames from {path.name} "
                f"using {successful_backend} backend"
            )
        except Exception as e:
            logger.error(f"Error while reading frames from {path.name}: {e}")
            raise ValueError(
                f"Failed to read frames from {path.name}: {e}\n"
                f"The video might be corrupted or in an unsupported format.\n"
                f"Try re-encoding: ffmpeg -i {path.name} -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p output.mp4"
            ) from e
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
        
        if not frames:
            raise ValueError(
                f"No frames read from {path.name}. Video might be empty or corrupted.\n"
                f"Check the file with: ffprobe {path.name}"
            )
        
        return frames

    def load_episode_frames(self, episode_index: int) -> tuple[list, str]:
        """Load raw frames and instruction for an episode without sampling or shuffling."""
        if episode_index < 0 or episode_index >= len(self.video_paths):
            raise IndexError
        video_path = self.video_paths[episode_index]
        logger.info(f"Loading MP4 episode {episode_index} from {video_path}")
        frames = self._load_video_frames(video_path)
        return frames, self.instruction

    def load_context_episodes(self, *, exclude_index: int | None = None) -> ContextEpisodes:
        if self.num_context_episodes <= 0:
            return ContextEpisodes([])
        pool = [
            idx for idx in range(len(self.video_paths))
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
            frames, instruction = self.load_episode_frames(idx)
            ctx_eps.append(self._build_episode(
                frames=frames,
                instruction=instruction,
                episode_index=idx,
                sampling_method=self.sampling_method,
                anchoring=self.anchoring,
            ))
        return ContextEpisodes(ctx_eps)

    def load_fewshot_input(self, episode_index: int | None = None) -> EpisodeEvalCase:
        if episode_index is None:
            episode_index = 0
        if episode_index < 0 or episode_index >= len(self.video_paths):
            raise IndexError
        frames, instruction = self.load_episode_frames(episode_index)
        ep = self._build_episode(
            frames=frames,
            instruction=instruction,
            episode_index=episode_index or 0,
            sampling_method=self.sampling_method,
            anchoring=self.anchoring,
        )
        context = self.load_context_episodes(exclude_index=episode_index)
        return EpisodeEvalCase(eval_episode=ep, context_episodes=context)
