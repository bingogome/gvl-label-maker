from gvl.clients.qwen25 import Qwen25Client

class MimoClient(Qwen25Client):
    def __init__(self, model_name: str = "XiaomiMiMo/MiMo-VL-7B-RL-2508", rpm: float = 0.0, max_input_length: int = 32768 ):
        super().__init__(model_name=model_name, rpm=rpm, max_input_length=max_input_length)