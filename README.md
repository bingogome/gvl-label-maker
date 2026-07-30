# Generative Value Learning Predictor

Inspired by [OpenGVL](https://github.com/budzianowski/opengvl), based on papers [OpenGVL](https://arxiv.org/abs/2509.17321) and [GVL](https://generative-value-learning.github.io/). This repo focuses on inferences instead of benchmarking.

## Examples

- Qwen3 8-bit: [demos/qwen3_8bit.mp4](demos/qwen3_8bit.mp4)
- Qwen3: [demos/qwen3.mp4](demos/qwen3.mp4)
- Gemma3 8-bit: [demos/gemma3_8bit.mp4](demos/gemma3_8bit.mp4)
- Gemini 2.5: [demos/gemini2_5.mp4](demos/gemini2_5.mp4)
- GPT-5.2: [demos/gpt5_2.mp4](demos/gpt5_2.mp4)
- Qwen3 with a phone recorded video: [demos/local.mp4](demos/local.mp4)
- Qwen3 with a phone recorded navigation video: [demos/nav.mp4](demos/nav.mp4)

<table>
  <tr>
    <th>Qwen/Qwen3-VL-32B-Instruct (Quantized to 8bit int)</th>
    <th>Qwen/Qwen3-VL-32B-Instruct</th>
    <th>gemma-3-27b-it (8bit)</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <a href="demos/qwen3_8bit.mp4">▶ Watch Qwen3 8-bit demo</a>
    </td>
    <td valign="top" width="33%">
      <a href="demos/qwen3.mp4">▶ Watch Qwen3 demo</a>
    </td>
    <td valign="top" width="33%">
      <a href="demos/gemma3_8bit.mp4">▶ Watch Gemma3 8-bit demo</a>
    </td>
  </tr>

  <tr>
    <th>gemini-2.5-pro</th>
    <th>gpt-5.2</th>
    <th>Qwen/Qwen3-VL-32B-Instruct — Phone Recording</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <a href="demos/gemini2_5.mp4">▶ Watch Gemini 2.5 Pro demo</a>
    </td>
    <td valign="top" width="33%">
      <a href="demos/gpt5_2.mp4">▶ Watch GPT-5.2 demo</a>
    </td>
    <td valign="top" width="33%">
      <a href="demos/local.mp4">▶ Watch phone-recorded demo</a>
    </td>
  </tr>

  <tr>
    <th>Qwen/Qwen3-VL-32B-Instruct — Navigation</th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <a href="demos/nav.mp4">▶ Watch navigation demo</a>
    </td>
    <td width="33%"></td>
    <td width="33%"></td>
  </tr>
</table>

## Quickstart Guide

```bash
# In the gvl-predictor repo
conda create -n gvl python=3.11.14
conda activate gvl
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install lerobot==0.4.1 --no-deps
```

Copy the `.env.example` to a new file and rename it to `.env`. Fill in the API and Tokens.

- `GEMINI_API_KEY` and `GOOGLE_API_KEY` are the same
    - required to use gemini
- `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are the same

```bash
# Run the following to test if the environment works
# Test 1 - results are stored in ./multirun folder
python -m gvl.scripts.benchmark --config-dir configs/experiments --config-name benchmark
# Test 2 - results are stored in ./results folder
python -m gvl.scripts.predict_episode_by_frame --config-dir configs/experiments --config-name test_predict_episode_by_frame
```

# Background

[**Generative Value Learning (GVL)**](https://arxiv.org/pdf/2411.04549)** is a universal value function estimator that leverages the world knowledge embedded in vision-language models (VLMs) to predict task progress.

[**Value-Order Correlation (VOC)**](https://arxiv.org/pdf/2411.04549) computes the rank correlation between the predicted values and the chronological order of the input expert video. 

- It ranges from -1 to 1 (perfect order).
- The repo implementation uses [Spearman’s rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

# Designed Behaviors

**Per Episode.** The benchmarking script runs on episodes (Behavior of OpenGVL). 

1. It packages a test episode and k context episode(s) as an evaluation case. 
2. Provides anchor frames (OpenGVL only supported single anchor)
3. Shuffles N sampled frames of the test episode and of each context episode(s)
4. Prompts a VLM with anchors and context frames
5. Prompts with the evaluation request on the test episode
6. Produce the progress scores of the N sampled frames of the test episode (minimally formatted raw text). 
7. Prompts a VLM (mapper) with the response to map the less-formatted response to structured JSON

**Per Frame.** The per episode behavior prompts the VLM *in batch*. Instead, the prediction script runs on frames (to supports online inference, not supported in OpenGVL). The prompting logic remains the same. Code correspondence details see Script Behaviors.

## Repo Structure

```
.
├── configs/
│   ├── data_loader/
│   ├── dataset/
│   ├── experiments/
│   ├── mapper/
│   ├── mapping_prompts/
│   ├── model/
│   ├── prompt_phrases/
│   └── prompts/
├── gvl/
│   ├── clients/
│   ├── data_loaders/
│   ├── mapper/
│   ├── metrics/
│   ├── results/
│   ├── scripts/
│   └── utils/
├── multirun/   # Hydra sweep outputs (generated)
├── results/    # Hydra run outputs (generated)
├── .env.example
├── requirements.txt
└── README.md
```

## Features

New:
- List of anchor frames: customizable anchors list, instead of only one frame.
- Conversation history logging with images and mapper outputs.
- Quantization
- Additional recent models
- Refactored evaluation templates. Independent eval and context. Optional frames and episodes eval. 
- Result frame-wise visualization, and optional labeled frames/videos
- Local video loader

Old:
- Data loading from LeRobot datasets and local image sequences, with context episodes, shuffling, and multiple sampling strategies.
- Prompt templates and prompt-phrase packs for different instruction styles and framing.
- Output parsing via regex or Gemini-based mapper with configurable mapping prompts.
- Metrics: Value-Order Correlation (VOC) for episode predictions and frame error for frame predictions.
- Outputs: streaming JSONL per prediction, summary JSON.

## Notebook Tutorial

Refer to 

- `notebooks/benchmark.ipynb` For batch benchmarking episodes.
- `notebooks/predict_episode_by_frames.ipynb` For predicting frame-by-frame.

## Script Behaviors

**Per Episode.** The benchmarking script runs on episodes (Behavior of OpenGVL). 

1. It packages a test episode and k context episode(s) as an evaluation case.
    - Built by `infer_utils.load_episode_eval_cases` in `gvl/utils/inference.py`, which calls `BaseDataLoader.load_fewshot_input` in `gvl/data_loaders/base.py` to produce an `EpisodeEvalCase` containing an `Episode` + `ContextEpisodes` from `gvl/utils/data_types.py`; triggered in `gvl/scripts/benchmark.py`.
2. Provides anchor frames (OpenGVL only supported single anchor).
    - Selected via `BaseDataLoader._select_anchor_frames` in `gvl/data_loaders/base.py`, stored in `Episode.anchor_frames`/`anchor_kinds` in `gvl/utils/data_types.py`, then emitted into the prompt by `BaseModelClient._iter_prompt_events` in `gvl/clients/base.py`.
3. Shuffles N sampled frames of the test episode and of each context episode(s).
    - Sampling via `BaseDataLoader._select_indices` and shuffling via `BaseDataLoader._maybe_shuffle` in `gvl/data_loaders/base.py`; stored as `Episode.shuffled_frames`/`Episode.shuffled_frames_indices` in `gvl/utils/data_types.py` for both eval and context episodes.
4. Prompts a VLM with anchors and context frames.
    - Prompt construction occurs in `BaseModelClient._iter_prompt_events` in `gvl/clients/base.py`; the base prompt string comes from `format_prompt` in `gvl/utils/prompts.py`; execution is via `BaseModelClient.generate_response_for_episode` in `gvl/clients/base.py`.
5. Prompts with the evaluation request on the test episode.
    - In `_iter_prompt_events`, it inserts `PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION` and labels eval frames using `PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE` in `gvl/clients/base.py`, iterating over `Episode.shuffled_frames` from `gvl/utils/data_types.py`.
6. Produce the progress scores of the N sampled frames of the test episode (less-formatted).
    - The raw VLM text is returned by `BaseModelClient._generate_from_events` via `BaseModelClient.generate_response_for_episode` in `gvl/clients/base.py`; parsing into `EpisodePredictionRecord.predicted_percentages` happens in `infer_utils._extract_percentages` in `gvl/utils/inference.py` and is stored in `gvl/results/prediction.py`.
7. Prompts a VLM (mapper) with the response to map the less-formatted response to parsed JSON.
    - Implemented by `BaseMapper.extract_percentages` (interface) and concretely by `GeminiMapper.extract_percentages` in `gvl/mapper/gemini_mapper.py`, invoked inside `infer_utils._extract_percentages` in `gvl/utils/inference.py`.

**Per Frame.** The per-episode behavior prompts the VLM *in batch*. Instead, the prediction script runs on frames (to supports online inference, not supported in OpenGVL). The prompting logic remains the same. 

1. It packages a test frame and k context episode(s) as an evaluation case.
    - The test frame is wrapped in `EvalFrame` and then `FrameEvalCase` (`gvl/utils/data_types.py`) inside `gvl/scripts/predict_frame.py`, using `BaseDataLoader.load_context_episodes` to build `ContextEpisodes` (`gvl/data_loaders/base.py`).
2. Provides anchor frames (OpenGVL only supported single anchor).
    - Anchor frames are chosen in `gvl/scripts/predict_frame.py` by `_select_anchor_frames` and attached as `EvalFrame.anchor_frames`/`anchor_kinds`; they are emitted into the prompt in `BaseModelClient._iter_prompt_events` (`gvl/clients/base.py`).
3. Shuffles N sampled frames of each context episode(s).
    - Context episodes are built by `BaseDataLoader.load_context_episodes`, which internally uses `_select_indices` and `_maybe_shuffle` in `gvl/data_loaders/base.py` to produce `Episode.shuffled_frames` and `Episode.shuffled_frames_indices` (`gvl/utils/data_types.py`).
4. Prompts a VLM with anchors and context frames.
    - Prompt assembly happens in `BaseModelClient._iter_prompt_events` (anchors + context frames) and is sent via `BaseModelClient.generate_response_for_frame` (`gvl/clients/base.py`), using the base prompt from `format_prompt` (`gvl/utils/prompts.py`).
5. Prompts with the evaluation request on the test frame.
    - `_iter_prompt_events` inserts `PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION` and labels the eval frame with `PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE` (`gvl/clients/base.py`).
6. Produce the progress scores of the test frame (minimally formatted raw text).
    - The raw model response is returned by `BaseModelClient._generate_from_events` through `BaseModelClient.generate_response_for_frame` (`gvl/clients/base.py`) and captured in `infer_utils._generate_eval_case_response` (`gvl/utils/inference.py`).

## Known Issues

```
ImportError("cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'")
```

Add the following two lines to `ENV_SITE_PACKAGES/transformers/utils/import_utils.py`
```
def is_torch_fx_available() -> bool:
    return is_torch_available()
```
