# orchestrator/nodes/generate.py
# Generate Node (SGLang Real Call)


from models.load_model import ModelWrapper

class GenerateNode:
    def __init__(self):
        self.model = ModelWrapper({
            "model_name": "deepseek-coder-v3",
            "cache_dir": "./models"
        })

    def run(self, task: str):
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "code": {"type": "string"}
            },
            "required": ["answer"]
        }

        result = self.model.generate(task, schema)

        return {
            "task": task,
            "output": result["output"],
            "valid": result["schema_valid"]
        }