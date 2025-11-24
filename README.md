# AI Code Improvement Playground

A safe, human-supervised AI system for code generation, analysis, and improvement using advanced language models with comprehensive safety mechanisms and enterprise-grade architecture.

## 🏗️ Architecture Overview

This system implements a controlled AI code improvement pipeline with the following key components:

### Core Components

- **Configuration Management**: Centralized, type-safe configuration with Pydantic validation
- **Orchestrator**: Central coordination system managing improvement cycles with safety limits
- **Processing Nodes**: Specialized components with consistent interfaces and error handling
- **Model Factory**: Singleton pattern for efficient model instance management
- **Security Layer**: Comprehensive input validation, sanitization, and security checks
- **Memory System**: LanceDB-based storage with semantic embeddings for episode logging
- **Testing Infrastructure**: Comprehensive test suite with fixtures and mocking

### System Flow

```
User Input → Security Validation → Orchestrator → [Generate → Reflect → Test → Learn] → Dashboard Output
                                      ↓
                                  Memory Store (LanceDB)
                                      ↓
                                  Model Factory (Cached Instances)
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

# Install Python dependencies
pip install -r requirements.txt

# Install static analysis tools
pip install ruff
npm install -g pyright eslint

# Create necessary directories
mkdir -p logs config/defaults
```

### Configuration Setup

Create configuration files in the `config/` directory:

```bash
# Copy default configurations
cp config/defaults/* config/
```

### Basic Usage

```bash
python main.py
```

The system provides an interactive CLI with the following features:
- **Task Processing**: Enter coding tasks for AI-assisted development
- **Safety Checks**: Comprehensive input validation and sanitization
- **Rate Limiting**: Prevents abuse with configurable limits
- **Statistics**: View system performance and usage metrics
- **Help System**: Built-in help and command reference

## 📁 Project Structure

```
playground/
├── main.py                    # Entry point with enhanced CLI and safety features
├── requirements.txt           # Python dependencies
├── config/                    # Configuration management
│   ├── config_manager.py     # Centralized configuration with Pydantic validation
│   ├── defaults/             # Default configuration templates
│   ├── model.yaml           # Model settings (DeepSeek Coder v3)
│   ├── cycles.yaml          # Safety limits and cycle configuration
│   └── tools.yaml           # Tool enablement/disablement
├── orchestrator/             # Core orchestration logic
│   ├── graph.py             # Main improvement graph coordinator
│   └── nodes/               # Processing nodes with consistent interfaces
│       ├── base.py          # Abstract base class for all nodes
│       ├── generate.py      # Code generation using SGLang + Model Factory
│       ├── reflect.py       # Self-reflection on outputs
│       ├── test.py          # Enhanced static analysis with security checks
│       └── learn.py         # Episode logging to memory
├── models/                  # Model loading and management
│   ├── load_model.py       # SGLang + TensorRT-LLM wrapper
│   └── model_factory.py    # Singleton pattern for model instance management
├── security/               # Security and validation layer
│   ├── __init__.py
│   └── validators.py       # Input validation, sanitization, and rate limiting
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── exceptions.py       # Custom exception hierarchy
│   └── logging_config.py   # Structured logging with security filtering
├── memory/                 # Memory and storage systems
│   └── memory.py          # LanceDB integration with embeddings
├── ui/                    # User interface components
│   └── dashboard/         # Console-based monitoring
│       └── monitor.py     # Enhanced result display
├── tests/                 # Comprehensive testing infrastructure
│   ├── __init__.py
│   ├── conftest.py        # Pytest configuration and fixtures
│   ├── test_config_manager.py  # Configuration management tests
│   ├── test_security.py   # Security validation tests
│   ├── test_nodes.py      # Node functionality tests
│   └── test_integration.py # Integration tests
├── finetune/             # Fine-tuning pipeline (human-gated)
│   └── trainer.py        # Dataset preparation and training scaffold
├── monetization/         # Commercial application scaffolds
│   ├── pr_agent/         # PR suggestion agent (safe-only)
│   └── vscode_extension/ # VS Code extension skeleton
└── logs/                 # Application logs (created at runtime)
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
5. **Input Validation**: Comprehensive sanitization and security pattern detection
6. **Rate Limiting**: Prevents abuse with configurable request limits
7. **Structured Logging**: Comprehensive audit trails with security filtering
8. **No Model Updates**: Learning logs data but doesn't modify model weights
9. **No External Actions**: Cannot submit PRs or modify external systems

## 🔧 Major Refactoring Improvements

This codebase has been significantly refactored to implement enterprise-grade architecture patterns:

### 1. Configuration Management
- **Implementation**: Centralized `ConfigManager` with Pydantic validation
- **Benefits**: Type-safe configuration, validation, caching, and hot-reload capability
- **Files**: `config/config_manager.py`, configuration models with validation

### 2. Base Node Architecture
- **Implementation**: Abstract `BaseNode` class with consistent interfaces
- **Benefits**: Standardized error handling, logging, metrics, and lifecycle management
- **Files**: `orchestrator/nodes/base.py`, refactored node implementations

### 3. Error Handling & Logging
- **Implementation**: Custom exception hierarchy and structured logging
- **Benefits**: Better error categorization, JSON logging, security filtering
- **Files**: `utils/exceptions.py`, `utils/logging_config.py`

### 4. Security Enhancements
- **Implementation**: Comprehensive input validation and sanitization
- **Benefits**: Protection against code injection, path traversal, and malicious input
- **Files**: `security/validators.py`, security pattern detection

### 5. Model Factory Pattern
- **Implementation**: Singleton pattern for expensive model instances
- **Benefits**: Resource efficiency, memory management, health monitoring
- **Files**: `models/model_factory.py`, cached model instances

### 6. Resource Management
- **Implementation**: Context managers and proper cleanup
- **Benefits**: Automatic resource cleanup, memory leak prevention
- **Files**: Enhanced temporary file handling, cleanup methods

### 7. Testing Infrastructure
- **Implementation**: Comprehensive pytest setup with fixtures
- **Benefits**: Reliable testing, mocking capabilities, coverage tracking
- **Files**: `tests/` directory with conftest.py and test modules

### 8. Type Safety
- **Implementation**: Comprehensive type hints and dataclasses
- **Benefits**: Better IDE support, runtime validation, documentation
- **Files**: Type annotations throughout codebase

### 9. Performance Optimizations
- **Implementation**: Efficient caching, lazy loading, resource pooling
- **Benefits**: Reduced memory usage, faster response times
- **Files**: Model factory caching, configuration caching

### 10. Enhanced CLI Interface
- **Implementation**: Rich interactive CLI with commands and safety features
- **Benefits**: Better user experience, built-in help, statistics
- **Files**: `main.py` with enhanced PlaygroundApp class

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