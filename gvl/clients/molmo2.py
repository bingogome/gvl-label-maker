from typing import cast

import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor

from gvl.clients.base import BaseModelClient
from gvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from gvl.utils.constants import MAX_TOKENS_TO_GENERATE
from gvl.utils.errors import InputTooLongError
from gvl.utils.images import to_pil


class Molmo2Client(BaseModelClient):
    """Client for Molmo2 image-text-to-text models."""

    def __init__(self, model_name: str = "allenai/Molmo2-8B", rpm: float = 0.0, max_input_length: int = 32768):
        super().__init__(rpm=rpm, max_input_length=max_input_length)
        logger.info(f"Loading Molmo2 model {model_name} ...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            token=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            token=True
        )
        self.model_name = model_name

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        if input_len > self.max_input_length:
            raise InputTooLongError(input_len, self.max_input_length)
        logger.info(f"Input length: {input_len}")

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS_TO_GENERATE,
                temperature=temperature,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=False)]
        return self.processor.tokenizer.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
