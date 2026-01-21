"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
import os
from pathlib import Path
from time import sleep
from typing import Any

from loguru import logger

from gvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from gvl.utils.constants import PromptPhraseKey
from gvl.utils.data_types import ContextEpisodes, Episode, EvalCase, EvalFrame, EpisodeEvalCase, FrameEvalCase
from gvl.utils.errors import MaxRetriesExceeded
from gvl.utils.images import to_pil
from gvl.utils.rate_limiter import SECS_PER_MIN, RateLimiter

MAX_RETRIES = 4  # how many times to retry on rate limit errors
PromptPhrases = dict[str, str | list[str]]


class BaseModelClient(ABC):
    """Abstract base class for all model clients.

    Subclasses must implement ``_generate_from_events``.
    They inherit image conversion & encoding helpers to produce standardized
    244x244 PNG base64 strings for multimodal APIs.
    """

    def __init__(self, *, rpm: float = 0.0, max_input_length: int = 32000) -> None:
        """Initialize the base model client.

        Args:
            rpm: Requests per minute rate limit (0.0 for no limit).
            max_input_length: Maximum input length for the model.
        """
        self.rpm = float(rpm)
        self.max_input_length = max_input_length
        # Persist a limiter instance so the rolling window spans calls.
        self._rate_limiter: RateLimiter | None = RateLimiter(max_calls=self.rpm, period=SECS_PER_MIN) if self.rpm > 0.0 else None
        log_dir = os.getenv("GVL_CONVERSATION_LOG_DIR")
        self._conversation_log_dir = Path(log_dir) if log_dir else None
        self._conversation_log_counter = 0
        self._last_conversation_log_path: Path | None = None

    def _run_with_rate_limit(self, fn: Callable[[], str]) -> str:
        if self._rate_limiter is None:
            return fn()
        logger.info(f"Applying rate limit: {self.rpm} requests per minute")
        with self._rate_limiter:
            logger.info("Lock acquired, generating response...")
            return fn()

    def _generate_with_retry(self, fn: Callable[[], str]) -> str:
        for call_attempt in range(1, MAX_RETRIES + 1):
            logger.debug(f"Model generation attempt {call_attempt}/{MAX_RETRIES}")
            try:
                res = self._run_with_rate_limit(fn)
                logger.info(f"Model response length: {len(res)} characters")
                return res
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Model generation attempt {call_attempt} failed: {e}")
                timesleep = 2 ** (call_attempt + 2)
                logger.warning(f"Retrying after {timesleep} seconds...")
                logger.error(f"Error details: {e}", exc_info=True) 
                sleep(timesleep)
        raise MaxRetriesExceeded(MAX_RETRIES)

    def generate_response_for_episode(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: ContextEpisodes,
        temperature: float = 0.0,
        *,
        prompt_phrases: PromptPhrases,
    ) -> str:
        """Generate a textual response for a given evaluation episode.

        This is the main entry point for generating model predictions.
        It wraps the subclass-specific implementation with rate limiting.

        Args:
            prompt: Base natural language instruction or system prompt.
            eval_episode: Episode whose frames require prediction.
            context_episodes: Few-shot context episodes with known progress.
        Returns:
            The raw model textual output.
        """
        return self._generate_with_prompt_phrases(
            prompt_phrases,
            lambda phrases: self._generate_response_for_episode_impl(
                prompt,
                eval_episode,
                context_episodes,
                temperature,
                prompt_phrases=phrases,
            ),
        )

    def generate_response_for_frame(
        self,
        prompt: str,
        eval_frame: EvalFrame,
        context_episodes: ContextEpisodes,
        temperature: float = 0.0,
        *,
        prompt_phrases: PromptPhrases,
    ) -> str:
        """Generate a textual response for a single evaluation frame."""
        return self._generate_with_prompt_phrases(
            prompt_phrases,
            lambda phrases: self._generate_response_for_frame_impl(
                prompt,
                eval_frame,
                context_episodes,
                temperature,
                prompt_phrases=phrases,
            ),
        )

    def generate_response_for_eval_case(
        self,
        prompt: str,
        eval_case: EvalCase,
        temperature: float = 0.0,
        *,
        prompt_phrases: PromptPhrases,
    ) -> str:
        """Generate a response for an EvalCase, dispatching by eval case type."""
        if isinstance(eval_case, EpisodeEvalCase):
            return self.generate_response_for_episode(
                prompt,
                eval_case.eval_episode,
                eval_case.context_episodes,
                temperature=temperature,
                prompt_phrases=prompt_phrases,
            )
        if isinstance(eval_case, FrameEvalCase):
            return self.generate_response_for_frame(
                prompt,
                eval_case.eval_frame,
                eval_case.context_episodes,
                temperature=temperature,
                prompt_phrases=prompt_phrases,
            )
        raise ValueError(f"Unsupported EvalCase type: {type(eval_case)}")

    def _generate_with_prompt_phrases(
        self,
        prompt_phrases: PromptPhrases,
        generate_fn: Callable[[PromptPhrases], str],
    ) -> str:
        """Normalize phrases and run the generation with retry handling."""
        phrases = self._validate_and_normalize_prompt_phrases(prompt_phrases)
        return self._generate_with_retry(lambda: generate_fn(phrases))

    def _validate_and_normalize_prompt_phrases(self, phrases: PromptPhrases) -> PromptPhrases:
        """Ensure all required phrase keys are present and non-empty strings.

        Returns a normalized dict that includes only the required keys.
        Raises ValueError if any required key is missing, not a string, or an empty string.
        Logs a debug message for any extra keys.
        """
        anchor_label_keys = [
            PromptPhraseKey.ANCHOR_SCENE_LABEL_START,
            PromptPhraseKey.ANCHOR_SCENE_LABEL_MIDDLE,
            PromptPhraseKey.ANCHOR_SCENE_LABEL_LAST,
        ]
        anchor_completion_keys = [
            PromptPhraseKey.ANCHOR_SCENE_COMPLETION_START,
            PromptPhraseKey.ANCHOR_SCENE_COMPLETION_MIDDLE,
            PromptPhraseKey.ANCHOR_SCENE_COMPLETION_LAST,
        ]
        required_keys = [
            *anchor_label_keys,
            *anchor_completion_keys,
            PromptPhraseKey.CONTEXT_FRAME_LABEL_TEMPLATE,
            PromptPhraseKey.CONTEXT_FRAME_COMPLETION_TEMPLATE,
            PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE,
            PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION,
        ]
        missing: list[str] = []
        normalized: PromptPhrases = {}
        for k in required_keys:
            key = k.value
            if key not in phrases:
                missing.append(key)
                continue
            value = phrases[key]
            if key == PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION.value:
                if isinstance(value, str):
                    instruction_list = [value] if value else []
                elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                    instruction_list = list(value)
                else:
                    instruction_list = []
                if not instruction_list or any(not isinstance(item, str) or not item for item in instruction_list):
                    missing.append(key)
                    continue
                normalized[key] = instruction_list
            else:
                if not isinstance(value, str) or not value:
                    missing.append(key)
                    continue
                normalized[key] = value

        if missing:
            raise ValueError("Missing or invalid (empty) prompt phrases for required keys: " + ", ".join(missing))

        # Log extra keys to help users diagnose config typos
        extras = [k for k in phrases if k not in normalized]
        if extras:
            logger.debug(f"Ignoring extra prompt phrase keys: {extras}")
        return normalized

    def _iter_prompt_events(
        self,
        prompt_text: str,
        *,
        instruction: str,
        anchor_frames: Sequence[ImageT] | None,
        anchor_kinds: Sequence[str] | None,
        eval_frames: Sequence[ImageT],
        context_episodes: ContextEpisodes,
        prompt_phrases: PromptPhrases,
    ) -> Iterator[Event]:
        phrases = prompt_phrases
        yield TextEvent(prompt_text)
        if anchor_frames:
            def _normalize_kind(kind: str) -> str:
                key = str(kind).strip().lower()
                if key == "first":
                    return "start"
                if key in {"start", "middle", "last"}:
                    return key
                raise ValueError(f"Unknown anchor kind: {kind}")

            def _default_kinds(count: int) -> list[str]:
                if count <= 0:
                    return []
                if count == 1:
                    return ["start"]
                if count == 2:
                    return ["start", "last"]
                return ["start"] + ["middle"] * (count - 2) + ["last"]

            resolved_kinds = [_normalize_kind(k) for k in anchor_kinds] if anchor_kinds else _default_kinds(len(anchor_frames))
            if len(resolved_kinds) != len(anchor_frames):
                raise ValueError("anchor_kinds length must match anchor_frames length")

            label_key_map = {
                "start": PromptPhraseKey.ANCHOR_SCENE_LABEL_START.value,
                "middle": PromptPhraseKey.ANCHOR_SCENE_LABEL_MIDDLE.value,
                "last": PromptPhraseKey.ANCHOR_SCENE_LABEL_LAST.value,
            }
            completion_key_map = {
                "start": PromptPhraseKey.ANCHOR_SCENE_COMPLETION_START.value,
                "middle": PromptPhraseKey.ANCHOR_SCENE_COMPLETION_MIDDLE.value,
                "last": PromptPhraseKey.ANCHOR_SCENE_COMPLETION_LAST.value,
            }

            for idx, (anchor, kind) in enumerate(zip(anchor_frames, resolved_kinds, strict=False), start=1):
                label_template = phrases[label_key_map[kind]]
                completion_template = phrases[completion_key_map[kind]]
                label = label_template.format(i=idx) if "{i}" in label_template else label_template
                completion = completion_template.format(i=idx) if "{i}" in completion_template else completion_template
                yield TextEvent(label)
                yield ImageEvent(anchor)
                yield TextEvent(completion)
        else:
            logger.debug("Missing anchor_frames; skipping initial scene anchor(s).")

        # Context frames (with known completion)
        counter = 1
        for ctx_episode in context_episodes:
            for task_completion, frame in zip(ctx_episode.shuffled_frames_approx_completion_rates, ctx_episode.shuffled_frames, strict=False):
                yield TextEvent(phrases[PromptPhraseKey.CONTEXT_FRAME_LABEL_TEMPLATE.value].format(i=counter))
                yield ImageEvent(frame)
                yield TextEvent(phrases[PromptPhraseKey.CONTEXT_FRAME_COMPLETION_TEMPLATE.value].format(p=task_completion))
                counter += 1

        num_frames = len(eval_frames)
        for instruction_str in phrases[PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION.value]:
            yield TextEvent(instruction_str.format(instruction=instruction, num_frames=num_frames))

        for frame in eval_frames:
            yield TextEvent(phrases[PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE.value].format(i=counter))
            yield ImageEvent(frame)
            yield TextEvent("")
            counter += 1

    def _generate_response_for_episode_impl(
        self,
        prompt: str,
        eval_episode: Episode,
        context_episodes: ContextEpisodes,
        temperature: float = 0.0,
        *,
        prompt_phrases: PromptPhrases,
    ) -> str:
        """Default implementation builds generic events and delegates to provider hook."""
        return self._generate_from_parts(
            prompt,
            instruction=eval_episode.instruction,
            anchor_frames=eval_episode.anchor_frames,
            anchor_kinds=eval_episode.anchor_kinds,
            eval_frames=eval_episode.shuffled_frames,
            context_episodes=context_episodes,
            temperature=temperature,
            prompt_phrases=prompt_phrases,
        )

    def _generate_response_for_frame_impl(
        self,
        prompt: str,
        eval_frame: EvalFrame,
        context_episodes: ContextEpisodes,
        temperature: float = 0.0,
        *,
        prompt_phrases: PromptPhrases,
    ) -> str:
        return self._generate_from_parts(
            prompt,
            instruction=eval_frame.instruction,
            anchor_frames=eval_frame.anchor_frames,
            anchor_kinds=eval_frame.anchor_kinds,
            eval_frames=[eval_frame.frame],
            context_episodes=context_episodes,
            temperature=temperature,
            prompt_phrases=prompt_phrases,
        )

    def _generate_from_parts(
        self,
        prompt: str,
        *,
        instruction: str,
        anchor_frames: Sequence[ImageT] | None,
        anchor_kinds: Sequence[str] | None,
        eval_frames: Sequence[ImageT],
        context_episodes: ContextEpisodes,
        temperature: float,
        prompt_phrases: PromptPhrases,
    ) -> str:
        events = list(
            self._iter_prompt_events(
                prompt,
                instruction=instruction,
                anchor_frames=anchor_frames,
                anchor_kinds=anchor_kinds,
                eval_frames=eval_frames,
                context_episodes=context_episodes,
                prompt_phrases=prompt_phrases,
            )
        )
        log_path = self._start_conversation_log(events)
        try:
            response_text = self._generate_from_events(events, temperature)
        except Exception as exc:
            self._finish_conversation_log(log_path, error=exc)
            raise
        self._finish_conversation_log(log_path, response_text=response_text)
        return response_text

    def _start_conversation_log(self, events: list[Event]) -> Path | None:
        if self._conversation_log_dir is None:
            return None
        try:
            self._conversation_log_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            model_name = getattr(self, "model_name", "model")
            model_safe = str(model_name).replace("/", "_")
            session_dir = self._conversation_log_dir / model_safe / f"{timestamp}_{self._conversation_log_counter:04d}"
            session_dir.mkdir(parents=True, exist_ok=True)
            log_path = session_dir / "conversation_history.txt"
            self._last_conversation_log_path = log_path
            os.environ["GVL_CONVERSATION_LOG_PATH"] = str(log_path)

            lines: list[str] = [
                f"model: {model_name}",
                f"timestamp: {timestamp}",
                "",
                "[PROMPT]",
                "",
            ]
            image_index = 0
            for idx, ev in enumerate(events, start=1):
                if isinstance(ev, TextEvent):
                    lines.append(f"[TEXT {idx:03d}]")
                    lines.append(ev.text)
                elif isinstance(ev, ImageEvent):
                    image_index += 1
                    image_name = f"image_{image_index:03d}.png"
                    image_path = session_dir / image_name
                    try:
                        to_pil(ev.image).save(image_path)
                        lines.append(f"[IMAGE {idx:03d}] {image_name}")
                    except Exception as exc:
                        lines.append(f"[IMAGE {idx:03d}] <save failed: {exc}>")
                else:
                    lines.append(f"[EVENT {idx:03d}] <unknown>")
                lines.append("")
            lines.append("[RESPONSE]")
            lines.append("")
            log_path.write_text("\n".join(lines), encoding="utf-8")
            return log_path
        except Exception as exc:
            logger.warning(f"Conversation logging failed: {exc}")
            return None

    def _finish_conversation_log(self, log_path: Path | None, *, response_text: str | None = None, error: Exception | None = None) -> None:
        if log_path is None:
            return
        try:
            with log_path.open("a", encoding="utf-8") as f:
                if error is not None:
                    f.write(f"<error: {error}>")
                elif response_text is not None:
                    f.write(response_text)
                f.write("\n")
        except Exception as exc:
            logger.warning(f"Conversation logging failed: {exc}")

    def _cleanup_handle(self, handle: Any, *, label: str) -> None:
        if handle is None:
            return
        for method_name in ("close", "shutdown", "cleanup", "release"):
            method = getattr(handle, method_name, None)
            if callable(method):
                try:
                    method()
                    logger.debug(f"Called {method_name} on {label}")
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug(f"{label}.{method_name} failed: {exc}")
                break

    def close(self) -> None:
        """Best-effort cleanup for clients and model resources."""
        self._cleanup_handle(getattr(self, "client", None), label="client")
        self._cleanup_handle(getattr(self, "model", None), label="model")
        self._cleanup_handle(getattr(self, "processor", None), label="processor")
        for attr in ("client", "model", "processor"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug(f"Failed to clear {attr}: {exc}")
        self._rate_limiter = None

    def __del__(self) -> None:  # pragma: no cover - best effort during GC
        try:
            self.close()
        except Exception:
            pass

    @abstractmethod
    def _generate_from_events(self, events: list[Event], temperature: float) -> str:  # pragma: no cover - interface only
        """Transform provider-agnostic prompt events into a model response."""
        raise NotImplementedError
