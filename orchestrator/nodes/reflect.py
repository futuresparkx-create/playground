# orchestrator/nodes/reflect.py
# Reflect Node (Real Implementation)

from models.load_model import ModelWrapper

class ReflectNode:
    def __init__(self):
        self.model = ModelWrapper({"model_name": "deepseek-coder-v3"})

    def run(self, text: str):
        return self.model.reflect(text)
