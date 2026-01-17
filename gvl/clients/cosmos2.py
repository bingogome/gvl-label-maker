from gvl.clients.qwen3 import Qwen3Client


class Cosmos2Client(Qwen3Client):
    def __init__(self, model_name: str = "nvidia/Cosmos-Reason2-8B", rpm: float = 0.0, max_input_length: int = 32768 ):
        super().__init__(model_name=model_name, rpm=rpm, max_input_length=max_input_length)