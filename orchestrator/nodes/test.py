# orchestrator/nodes/test.py
# Real Static Analysis Node

import subprocess
import tempfile
import uuid
import os

class TestNode:
    """
    Static analysis for safety.
    Never executes the code.
    """

    def run(self, code: str):
        lang = self.detect_language(code)

        if not lang:
            return {
                "static_result": "unknown language",
                "errors": [],
                "warnings": []
            }

        if lang == "python":
            return self.test_python(code)

        if lang in {"javascript", "typescript"}:
            return self.test_js(code)

        return {
            "static_result": "unsupported language",
            "errors": [],
            "warnings": []
        }

    def detect_language(self, code: str):
        if "def " in code or "import " in code:
            return "python"
        if "function " in code or "const " in code:
            return "javascript"
        return None

    def test_python(self, code: str):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = f"{tmp}/{uuid.uuid4().hex}.py"

            with open(file_path, "w") as f:
                f.write(code)

            # Ruff linter
            ruff = subprocess.run(
                ["ruff", "check", file_path, "--format=json"],
                capture_output=True, text=True
            )

            # Pyright type checker
            pyright = subprocess.run(
                ["pyright", file_path, "--outputjson"],
                capture_output=True, text=True
            )

            return {
                "static_result": "python_analysis",
                "errors": self.parse(ruff.stdout) + self.parse(pyright.stdout),
                "warnings": []
            }

    def test_js(self, code: str):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = f"{tmp}/{uuid.uuid4().hex}.js"
            with open(file_path, "w") as f:
                f.write(code)

            eslint = subprocess.run(
                ["eslint", file_path, "-f", "json"],
                capture_output=True, text=True
            )

            return {
                "static_result": "js_analysis",
                "errors": self.parse(eslint.stdout),
                "warnings": []
            }

    def parse(self, output: str):
        try:
            import json
            return json.loads(output)
        except:
            return []