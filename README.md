# playground
# AI Code Improvement Playground

A safe, human-supervised AI system for code generation, analysis, and improvement using advanced language models with built-in safety mechanisms.

## 🏗️ Architecture Overview

This system implements a controlled AI code improvement pipeline with the following key components:

### Core Components

- **Orchestrator**: Central coordination system managing improvement cycles with safety limits
- **Processing Nodes**: Specialized components for generation, reflection, testing, and learning
- **Memory System**: LanceDB-based storage with semantic embeddings for episode logging
- **Safety Layer**: Multiple safeguards preventing autonomous execution and ensuring human oversight

### System Flow

```
User Input → Orchestrator → [Generate → Reflect → Test → Learn] → Dashboard Output
                ↓
            Memory Store (LanceDB)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- SGLang with TensorRT-LLM backend
- Required dependencies (see Installation section)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd playground

# Install dependencies
pip install sgllm lancedb sentence-transformers

# Install static analysis tools
pip install ruff pyright
npm install -g eslint
```

### Basic Usage

```bash
python main.py
```

Enter a coding task when prompted, and the system will:
1. Generate a solution using DeepSeek Coder v3
2. Perform self-reflection on the output
3. Run static analysis (no code execution)
4. Log the episode to memory
5. Display results via dashboard

## 📁 Project Structure

```
playground/
├── main.py                 # Entry point and main orchestration
├── config/                 # Configuration files
│   ├── model.yaml         # Model settings (DeepSeek Coder v3)
│   ├── cycles.yaml        # Safety limits and cycle configuration
│   └── tools.yaml         # Tool enablement/disablement
├── orchestrator/          # Core orchestration logic
│   ├── graph.py          # Main improvement graph coordinator
│   └── nodes/            # Processing nodes
│       ├── generate.py   # Code generation using SGLang
│       ├── reflect.py    # Self-reflection on outputs
│       ├── test.py       # Static analysis (ruff, pyright, eslint)
│       └── learn.py      # Episode logging to memory
├── models/               # Model loading and inference
│   └── load_model.py    # SGLang + TensorRT-LLM wrapper
├── memory/              # Memory and storage systems
│   └── memory.py       # LanceDB integration with embeddings
├── ui/                 # User interface components
│   └── dashboard/      # Console-based monitoring
│       └── monitor.py  # Simple result display
├── finetune/          # Fine-tuning pipeline (human-gated)
│   └── trainer.py     # Dataset preparation and training scaffold
└── monetization/      # Commercial application scaffolds
    ├── pr_agent/      # PR suggestion agent (safe-only)
    └── vscode_extension/  # VS Code extension skeleton
```

## ⚙️ Configuration

### Model Configuration (`config/model.yaml`)
```yaml
model:
  name: deepseek_coder_v3
  engine: sgllang_trt
  max_tokens: 8192
  temperature: 0.1
  top_p: 0.95
```

### Safety Configuration (`config/cycles.yaml`)
```yaml
cycles:
  max_cycles: 10              # Maximum improvement cycles
  max_reflect: 3              # Maximum reflection iterations
  require_human_approval: true # Human oversight required
```

### Tool Configuration (`config/tools.yaml`)
```yaml
tools:
  enable:
    - code_generation
    - static_analysis
    - reflection
    - dataset_logging
  disabled:
    - autonomous_execution     # Safety: No code execution
    - remote_actions          # Safety: No external actions
    - weight_updates          # Safety: No model modifications
```

## 🛡️ Safety Features

This system is designed with multiple safety layers:

1. **No Autonomous Execution**: Generated code is never executed automatically
2. **Human Approval Gates**: Manual cycle invocation required
3. **Bounded Cycles**: Configurable limits on improvement iterations
4. **Static Analysis Only**: Code quality checks without execution
5. **No Model Updates**: Learning logs data but doesn't modify model weights
6. **No External Actions**: Cannot submit PRs or modify external systems

## 🔧 Advanced Usage

### Memory Queries
The system stores episodes in LanceDB with semantic embeddings:

```python
from memory.memory import MemoryStore

memory = MemoryStore()
results = memory.fetch_semantic("error handling patterns", k=5)
```

### Custom Node Development
Extend the system by creating new processing nodes:

```python
from orchestrator.nodes.base import BaseNode

class CustomNode(BaseNode):
    def run(self, input_data):
        # Your custom processing logic
        return {"result": "processed"}
```

## 🚧 Development Status

### Implemented
- ✅ Core orchestration system
- ✅ SGLang + TensorRT-LLM integration
- ✅ Static analysis pipeline
- ✅ LanceDB memory system
- ✅ Safety mechanisms

### In Development
- 🔄 Fine-tuning pipeline
- 🔄 PR agent capabilities
- 🔄 VS Code extension
- 🔄 Advanced dashboard UI

## 🤝 Contributing

1. Ensure all safety mechanisms remain intact
2. Add tests for new functionality
3. Update documentation for new features
4. Follow the existing code structure and patterns

## 📄 License

[Add your license information here]

## 🔗 Related Projects

- [SGLang](https://github.com/sgl-project/sglang) - Structured Generation Language
- [LanceDB](https://github.com/lancedb/lancedb) - Vector database
- [DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder) - Code generation model
