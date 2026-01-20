from typing import cast

import torch
from loguru import logger
from torchao.quantization import Int4WeightOnlyConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, TorchAoConfig

from gvl.clients.base import BaseModelClient
from gvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from gvl.utils.constants import MAX_TOKENS_TO_GENERATE
from gvl.utils.images import to_pil
from qwen_vl_utils import process_vision_info


class Qwen4BitClient(BaseModelClient):
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", rpm: float = 0.0, max_input_length: int = 32768 ):
        super().__init__(rpm=rpm, max_input_length=max_input_length)
        self.model_name = model_name
        self.max_input_length = max_input_length
    
        quantization_config = TorchAoConfig(
            Int4WeightOnlyConfig(set_inductor_config=False)
        )

        if "qwen3" in model_name.lower() and "a3b" in model_name.lower():
            logger.info(f"Loading Qwen3 model {model_name} ...")
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )
        elif "qwen3" in model_name.lower() and "a3b" not in model_name.lower():
            logger.info(f"Loading Qwen3 model {model_name} ...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            logger.info(f"Loading Qwen3-type model {model_name} ...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.info(type(self.processor))
    
    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        input_len = inputs["input_ids"].shape[-1]
        if input_len > self.max_input_length:
            raise ValueError(f"Input length {input_len} exceeds maximum allowed {self.max_input_length}")
        logger.info(f"Input length: {input_len}")

        # Inference: Generation of the output
        if temperature == 0.0:
            generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, do_sample=False)
        else:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS_TO_GENERATE,
                do_sample=True,
                temperature=temperature
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
