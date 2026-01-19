# Generative Value Learning Label Maker

Inspired by [OpenGVL](https://github.com/budzianowski/opengvl), based on papers [OpenGVL](https://arxiv.org/abs/2509.17321) and [GVL](https://generative-value-learning.github.io/). This repo focuses on inferences instead of benchmarking.

## Features

New:
- Anchor frames: start/middle/last with customizable anchor labels/completions in prompt phrases.
- Conversation history logging with images and mapper outputs.
- Quantization
- Additional recent models
- Refactored evaluation templates. Independent eval and context. Optional frames and episodes eval. 
- Result frame-wise visualization

Old:
- Data loading from LeRobot datasets and local image sequences, with context episodes, shuffling, and multiple sampling strategies.
- Supported models: OpenAI, Gemini, Gemma (full/4bit/8bit), Qwen2.5/Qwen3 (full/quantized), GLM, Cosmos, Mimo, Molmo2.
- Prompt templates and prompt-phrase packs for different instruction styles and framing.
- Output parsing via regex or Gemini-based mapper with configurable mapping prompts.
- Metrics: Value-Order Correlation (VOC) for episode predictions and frame error for frame predictions.
- Outputs: streaming JSONL per prediction, summary JSON, and optional labeled frames/videos.

## Known Issues

```
ImportError("cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'")
```

Add the following two lines to `ENV_SITE_PACKAGES/transformers/utils/import_utils.py`
```
def is_torch_fx_available() -> bool:
    return is_torch_available()
```