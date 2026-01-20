"""GGUF client backed by llama.cpp (llama-cpp-python).

This client supports local GGUF files and optional Hugging Face downloads.
For vision models, provide an mmproj path and a compatible chat handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from loguru import logger

from gvl.clients.base import BaseModelClient
from gvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from gvl.utils.constants import MAX_TOKENS_TO_GENERATE
from gvl.utils.images import encode_image


class GGUFClient(BaseModelClient):
    """Client for GGUF models via llama-cpp-python."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        mmproj_path: str | None = None,
        mmproj_repo_id: str | None = None,
        mmproj_filename: str | None = None,
        chat_format: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        rpm: float = 0.0,
    ) -> None:
        super().__init__(rpm=rpm)
        model_path_resolved = self._resolve_path(
            model_path=model_path,
            repo_id=repo_id,
            filename=filename,
            label="model",
            required=True,
        )
        mmproj_path_resolved = self._resolve_path(
            model_path=mmproj_path,
            repo_id=mmproj_repo_id,
            filename=mmproj_filename,
            label="mmproj",
            required=False,
        )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("GGUFClient requires llama-cpp-python to be installed.") from exc

        chat_handler = None
        if mmproj_path_resolved is not None:
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
            except ImportError as exc:
                raise ImportError("llama-cpp-python with chat handlers is required for vision GGUF models.") from exc
            chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path_resolved))
            self._supports_vision = True
        else:
            self._supports_vision = False

        logger.info(f"Loading GGUF model from {model_path_resolved}")
        self.model_name = str(model_path_resolved)
        self.llm = Llama(
            model_path=str(model_path_resolved),
            chat_format=chat_format,
            chat_handler=chat_handler,
            n_ctx=int(n_ctx),
            n_gpu_layers=int(n_gpu_layers),
            n_threads=int(n_threads) if n_threads is not None else None,
            verbose=False,
        )

    def _resolve_path(
        self,
        *,
        model_path: str | None,
        repo_id: str | None,
        filename: str | None,
        label: str,
        required: bool,
    ) -> Path | None:
        if model_path:
            return Path(model_path)
        if repo_id and filename:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError("huggingface_hub is required to download GGUF files by repo_id.") from exc
            return Path(hf_hub_download(repo_id=repo_id, filename=filename))
        if repo_id and not filename:
            raise ValueError(f"{label}_filename is required when {label}_repo_id is provided.")
        if required:
            raise ValueError(f"Provide {label}_path or {label}_repo_id + {label}_filename.")
        return None

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        has_images = any(isinstance(ev, ImageEvent) for ev in events)
        if not has_images:
            text = "\n".join(ev.text for ev in events if isinstance(ev, TextEvent))
            messages = [{"role": "user", "content": text}]
        else:
            messages = [{"role": "user", "content": []}]
            for ev in events:
                if isinstance(ev, TextEvent):
                    messages[0]["content"].append({"type": "text", "text": ev.text})
                elif isinstance(ev, ImageEvent):
                    if not self._supports_vision:
                        raise ValueError("GGUF model is not configured for vision; provide mmproj_path.")
                    img = cast(ImageT, ev.image)
                    b64 = encode_image(img)
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS_TO_GENERATE,
        )
        return response["choices"][0]["message"]["content"]
