# tests/conftest.py
"""
Pytest configuration and fixtures for the playground test suite.
Provides common test fixtures and configuration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from config.config_manager import ConfigManager, ModelConfig, CyclesConfig, ToolsConfig
from orchestrator.nodes.base import NodeConfig
from utils.logging_config import setup_logging


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    setup_logging(log_level="DEBUG", json_format=False)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "model": ModelConfig(
            name="test-model",
            engine="mock",
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95
        ),
        "cycles": CyclesConfig(
            max_cycles=5,
            max_reflect=2,
            require_human_approval=False  # Disable for testing
        ),
        "tools": ToolsConfig(
            enable=["code_generation", "static_analysis", "reflection"],
            disabled=["autonomous_execution", "remote_actions"]
        )
    }


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.generate.return_value = {
        "output": {
            "answer": "Test solution explanation",
            "code": "def test_function():\n    return 'Hello, World!'",
            "language": "python"
        },
        "schema_valid": True,
        "generation_time": 0.5
    }
    model.health_check.return_value = True
    model.cleanup = Mock()
    return model


@pytest.fixture
def mock_model_factory(mock_model):
    """Mock model factory for testing."""
    factory = Mock()
    factory.create_model.return_value = mock_model
    factory.get_model_info.return_value = {}
    factory.cleanup_all = Mock()
    return factory


@pytest.fixture
def node_config():
    """Default node configuration for testing."""
    return NodeConfig(
        enabled=True,
        timeout=30.0,
        retry_count=1,
        retry_delay=0.1,
        validate_input=True,
        validate_output=True
    )


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    print(fibonacci(10))

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return """
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function main() {
    console.log(fibonacci(10));
}

main();
"""


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return "Create a function that calculates the fibonacci sequence"


@pytest.fixture
def config_manager_with_temp_dir(temp_dir):
    """ConfigManager with temporary directory."""
    config_dir = temp_dir / "config"
    config_dir.mkdir()
    
    # Create test config files
    model_config = {
        "model": {
            "name": "test-model",
            "engine": "mock",
            "max_tokens": 1000
        }
    }
    
    cycles_config = {
        "cycles": {
            "max_cycles": 5,
            "max_reflect": 2,
            "require_human_approval": False
        }
    }
    
    tools_config = {
        "tools": {
            "enable": ["code_generation", "static_analysis"],
            "disabled": ["autonomous_execution"]
        }
    }
    
    import yaml
    
    with open(config_dir / "model.yaml", "w") as f:
        yaml.dump(model_config, f)
    
    with open(config_dir / "cycles.yaml", "w") as f:
        yaml.dump(cycles_config, f)
    
    with open(config_dir / "tools.yaml", "w") as f:
        yaml.dump(tools_config, f)
    
    return ConfigManager(config_dir)


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for testing static analysis tools."""
    def _mock_run(cmd, **kwargs):
        result = Mock()
        result.returncode = 0
        result.stdout = "[]"  # Empty JSON array
        result.stderr = ""
        return result
    
    return _mock_run


@pytest.fixture
def security_test_cases():
    """Test cases for security validation."""
    return {
        "safe_code": "def hello():\n    return 'Hello, World!'",
        "dangerous_eval": "eval('print(\"dangerous\")')",
        "dangerous_exec": "exec('import os; os.system(\"ls\")')",
        "dangerous_import": "import os; os.system('rm -rf /')",
        "file_operation": "with open('/etc/passwd', 'r') as f:\n    content = f.read()",
        "network_operation": "import requests; requests.get('http://evil.com')"
    }


@pytest.fixture
def generation_test_cases():
    """Test cases for code generation."""
    return {
        "simple_function": "Create a function that adds two numbers",
        "class_definition": "Create a class for managing a shopping cart",
        "algorithm": "Implement a binary search algorithm",
        "data_structure": "Create a linked list implementation",
        "web_scraping": "Create a function to scrape data from a website",
        "file_processing": "Create a function to process CSV files"
    }


@pytest.fixture
def analysis_test_cases():
    """Test cases for static analysis."""
    return {
        "valid_python": """
def calculate_sum(a, b):
    return a + b

result = calculate_sum(5, 3)
print(result)
""",
        "invalid_python": """
def calculate_sum(a, b)
    return a + b  # Missing colon

result = calculate_sum(5, 3
print(result)  # Missing closing parenthesis
""",
        "valid_javascript": """
function calculateSum(a, b) {
    return a + b;
}

const result = calculateSum(5, 3);
console.log(result);
""",
        "invalid_javascript": """
function calculateSum(a, b) {
    return a + b  // Missing semicolon
}

const result = calculateSum(5, 3)
console.log(result  // Missing closing parenthesis
"""
    }


@pytest.fixture
def memory_test_data():
    """Test data for memory operations."""
    return [
        {
            "task": "Create a sorting algorithm",
            "solution": "def bubble_sort(arr): ...",
            "metadata": {"language": "python", "complexity": "O(n^2)"}
        },
        {
            "task": "Implement a hash table",
            "solution": "class HashTable: ...",
            "metadata": {"language": "python", "data_structure": "hash_table"}
        },
        {
            "task": "Create a REST API endpoint",
            "solution": "@app.route('/api/users') ...",
            "metadata": {"language": "python", "framework": "flask"}
        }
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "test_security" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)