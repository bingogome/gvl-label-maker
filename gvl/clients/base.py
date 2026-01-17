"""Core base model client abstraction and image utilities.

This module intentionally keeps a very small surface area and avoids pulling heavy
dependencies (transformers, torch, cloud SDKs) so importing it is cheap for all
downstream modules.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from time import sleep

from loguru import logger

from gvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from gvl.utils.constants import PromptPhraseKey
from gvl.utils.data_types import ContextEpisodes, Episode, EvalCase, EvalFrame, EpisodeEvalCase, FrameEvalCase
from gvl.utils.errors import MaxRetriesExceeded
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
        required_keys = [
            PromptPhraseKey.INITIAL_SCENE_LABEL,
            PromptPhraseKey.INITIAL_SCENE_COMPLETION,
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
        starting_frame: ImageT | None,
        eval_frames: Sequence[ImageT],
        context_episodes: ContextEpisodes,
        prompt_phrases: PromptPhrases,
    ) -> Iterator[Event]:
        phrases = prompt_phrases
        yield TextEvent(prompt_text)
        if starting_frame is not None:
            yield TextEvent(phrases[PromptPhraseKey.INITIAL_SCENE_LABEL.value])
            yield ImageEvent(starting_frame)
            yield TextEvent(phrases[PromptPhraseKey.INITIAL_SCENE_COMPLETION.value])
        else:
            logger.debug("Missing starting_frame; skipping initial scene anchor.")

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
            starting_frame=eval_episode.starting_frame,
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
            starting_frame=eval_frame.starting_frame,
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
        starting_frame: ImageT | None,
        eval_frames: Sequence[ImageT],
        context_episodes: ContextEpisodes,
        temperature: float,
        prompt_phrases: PromptPhrases,
    ) -> str:
        events = list(
            self._iter_prompt_events(
                prompt,
                instruction=instruction,
                starting_frame=starting_frame,
                eval_frames=eval_frames,
                context_episodes=context_episodes,
                prompt_phrases=prompt_phrases,
            )
        )
        return self._generate_from_events(events, temperature)

    @abstractmethod
    def _generate_from_events(self, events: list[Event], temperature: float) -> str:  # pragma: no cover - interface only
        """Transform provider-agnostic prompt events into a model response."""
        raise NotImplementedError
