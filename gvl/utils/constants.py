from enum import StrEnum
from typing import Final

IMG_SIZE: Final[int] = 244
# MAX_TOKENS_TO_GENERATE: Final[int] = 1024
MAX_TOKENS_TO_GENERATE: Final[int] = 10000
N_DEBUG_PROMPT_CHARS: Final[int] = 400


class PromptPhraseKey(StrEnum):
    ANCHOR_SCENE_LABEL_START = "anchor_scene_label_start"
    ANCHOR_SCENE_LABEL_MIDDLE = "anchor_scene_label_middle"
    ANCHOR_SCENE_LABEL_LAST = "anchor_scene_label_last"
    ANCHOR_SCENE_COMPLETION_START = "anchor_scene_completion_start"
    ANCHOR_SCENE_COMPLETION_MIDDLE = "anchor_scene_completion_middle"
    ANCHOR_SCENE_COMPLETION_LAST = "anchor_scene_completion_last"
    CONTEXT_FRAME_LABEL_TEMPLATE = "context_frame_label_template"
    CONTEXT_FRAME_COMPLETION_TEMPLATE = "context_frame_completion_template"
    EVAL_FRAME_LABEL_TEMPLATE = "eval_frame_label_template"
    EVAL_TASK_COMPLETION_INSTRUCTION = "eval_task_completion_instruction"
