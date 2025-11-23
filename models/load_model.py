# models/load_model.py
# Real Model Loader (SGLang + TensorRT-LLM)

import sgllm
from sgllm import SGLang
from sgllm.schema import JsonResponse
from pathlib import Path

class ModelWrapper:
    """
    Safe inference wrapper around SGLang + TensorRT LLM backends.
    Only text in, text out. Schema validation optional.
    No code execution or self-modification.
    """

    def __init__(self, config):
        self.config = config

        model_name = config.get("model_name", "deepseek-coder-v3")
        cache_dir = config.get("cache_dir", "./models")

        self.engine = SGLang(
            model=model_name,
            backend="tensorrt-llm",
            quantization="fp8",
            max_batch_size=4,
            tensor_parallel=1,
            download_dir=cache_dir
        )

    def generate(self, prompt: str, schema: dict | None = None) -> dict:
        """
        Text generation with optional JSON schema validation using SGLang.
        """
        if schema:
            resp = self.engine.generate(
                prompt,
                response_format=JsonResponse(schema=schema),
                temperature=0.1,
                max_tokens=4096
            )
            return {
                "output": resp.text,
                "schema_valid": resp.is_valid
            }
        else:
            text = self.engine.generate(prompt, temperature=0.1)
            return {
                "output": text,
                "schema_valid": True
            }

    def reflect(self, text: str) -> str:
        """
        Reflection: short single-shot prompt.
        """
        reflection_prompt = (
            "Reflect on the following model output. "
            "Identify errors, ambiguities, or suboptimal reasoning.\n\n"
            f"OUTPUT:\n{text}"
        )
        return self.engine.generate(reflection_prompt, temperature=0.1)