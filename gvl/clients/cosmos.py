from gvl.clients.qwen25 import Qwen25Client


class CosmosClient(Qwen25Client):
    def __init__(self, model_name: str = "nvidia/Cosmos-Reason1-7B", rpm: float = 0.0, max_input_length: int = 32768 ):
        super().__init__(model_name=model_name, rpm=rpm, max_input_length=max_input_length)