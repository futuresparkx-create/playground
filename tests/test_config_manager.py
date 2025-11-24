# tests/test_config_manager.py
"""
Tests for the configuration management system.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from config.config_manager import ConfigManager, ModelConfig, CyclesConfig, ToolsConfig, ConfigurationError


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_load_config_with_valid_files(self, config_manager_with_temp_dir):
        """Test loading configuration with valid files."""
        config_manager = config_manager_with_temp_dir
        config = config_manager.load_config()
        
        assert "model" in config
        assert "cycles" in config
        assert "tools" in config
        
        assert isinstance(config["model"], ModelConfig)
        assert isinstance(config["cycles"], CyclesConfig)
        assert isinstance(config["tools"], ToolsConfig)
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid config
        valid_config = ModelConfig(
            name="test-model",
            engine="sgllang",
            max_tokens=8192,
            temperature=0.1,
            top_p=0.95
        )
        assert valid_config.name == "test-model"
        assert valid_config.max_tokens == 8192
        
        # Invalid temperature
        with pytest.raises(ValueError):
            ModelConfig(
                name="test-model",
                engine="sgllang",
                temperature=3.0  # Too high
            )
        
        # Invalid max_tokens
        with pytest.raises(ValueError):
            ModelConfig(
                name="test-model",
                engine="sgllang",
                max_tokens=0  # Too low
            )
    
    def test_cycles_config_validation(self):
        """Test cycles configuration validation."""
        # Valid config
        valid_config = CyclesConfig(
            max_cycles=10,
            max_reflect=3,
            require_human_approval=True
        )
        assert valid_config.max_cycles == 10
        assert valid_config.require_human_approval is True
        
        # Invalid max_cycles
        with pytest.raises(ValueError):
            CyclesConfig(max_cycles=0)  # Too low
        
        # Invalid max_reflect
        with pytest.raises(ValueError):
            CyclesConfig(max_reflect=-1)  # Negative
    
    def test_config_caching(self, config_manager_with_temp_dir):
        """Test configuration caching behavior."""
        config_manager = config_manager_with_temp_dir
        
        # First load
        config1 = config_manager.load_config()
        
        # Second load should return cached version
        config2 = config_manager.load_config()
        
        assert config1 is config2  # Same object reference
    
    def test_config_reload(self, config_manager_with_temp_dir):
        """Test configuration reload functionality."""
        config_manager = config_manager_with_temp_dir
        
        # Load initial config
        config1 = config_manager.load_config()
        
        # Reload config
        config_manager.reload_config()
        config2 = config_manager.load_config()
        
        # Should be different object references
        assert config1 is not config2
    
    def test_missing_config_files(self, temp_dir):
        """Test behavior with missing configuration files."""
        config_manager = ConfigManager(temp_dir / "nonexistent")
        
        with pytest.raises(ConfigurationError):
            config_manager.load_config()
    
    def test_invalid_yaml_format(self, temp_dir):
        """Test handling of invalid YAML format."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        
        # Create invalid YAML file
        with open(config_dir / "model.yaml", "w") as f:
            f.write("invalid: yaml: content: [")
        
        config_manager = ConfigManager(config_dir)
        
        with pytest.raises(ConfigurationError):
            config_manager.load_config()
    
    def test_get_specific_configs(self, config_manager_with_temp_dir):
        """Test getting specific configuration sections."""
        config_manager = config_manager_with_temp_dir
        
        model_config = config_manager.get_model_config()
        cycles_config = config_manager.get_cycles_config()
        tools_config = config_manager.get_tools_config()
        
        assert isinstance(model_config, ModelConfig)
        assert isinstance(cycles_config, CyclesConfig)
        assert isinstance(tools_config, ToolsConfig)
        
        assert model_config.name == "test-model"
        assert cycles_config.max_cycles == 5
    
    def test_validate_config_files(self, config_manager_with_temp_dir):
        """Test configuration file validation."""
        config_manager = config_manager_with_temp_dir
        
        # Should validate successfully
        assert config_manager.validate_config_files() is True
    
    def test_default_values(self, temp_dir):
        """Test default values when config files are missing."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        
        config_manager = ConfigManager(config_dir)
        
        # Should create default configurations
        model_config = config_manager.get_model_config()
        cycles_config = config_manager.get_cycles_config()
        
        # Check default values
        assert model_config.max_tokens == 8192
        assert model_config.temperature == 0.1
        assert cycles_config.max_cycles == 10
        assert cycles_config.require_human_approval is True


class TestModelConfig:
    """Test cases for ModelConfig."""
    
    def test_valid_model_config(self):
        """Test valid model configuration."""
        config = ModelConfig(
            name="deepseek-coder-v3",
            engine="sgllang_trt",
            max_tokens=4096,
            temperature=0.2,
            top_p=0.9
        )
        
        assert config.name == "deepseek-coder-v3"
        assert config.engine == "sgllang_trt"
        assert config.max_tokens == 4096
        assert config.temperature == 0.2
        assert config.top_p == 0.9
    
    def test_model_config_defaults(self):
        """Test model configuration defaults."""
        config = ModelConfig(name="test", engine="test")
        
        assert config.max_tokens == 8192
        assert config.temperature == 0.1
        assert config.top_p == 0.95
    
    def test_model_config_validation_ranges(self):
        """Test model configuration validation ranges."""
        # Valid ranges
        ModelConfig(name="test", engine="test", max_tokens=1)
        ModelConfig(name="test", engine="test", max_tokens=32768)
        ModelConfig(name="test", engine="test", temperature=0.0)
        ModelConfig(name="test", engine="test", temperature=2.0)
        ModelConfig(name="test", engine="test", top_p=0.0)
        ModelConfig(name="test", engine="test", top_p=1.0)
        
        # Invalid ranges
        with pytest.raises(ValueError):
            ModelConfig(name="test", engine="test", max_tokens=0)
        
        with pytest.raises(ValueError):
            ModelConfig(name="test", engine="test", max_tokens=50000)
        
        with pytest.raises(ValueError):
            ModelConfig(name="test", engine="test", temperature=-0.1)
        
        with pytest.raises(ValueError):
            ModelConfig(name="test", engine="test", temperature=2.1)


class TestCyclesConfig:
    """Test cases for CyclesConfig."""
    
    def test_valid_cycles_config(self):
        """Test valid cycles configuration."""
        config = CyclesConfig(
            max_cycles=5,
            max_reflect=2,
            require_human_approval=False
        )
        
        assert config.max_cycles == 5
        assert config.max_reflect == 2
        assert config.require_human_approval is False
    
    def test_cycles_config_defaults(self):
        """Test cycles configuration defaults."""
        config = CyclesConfig()
        
        assert config.max_cycles == 10
        assert config.max_reflect == 3
        assert config.require_human_approval is True
    
    def test_cycles_config_validation_ranges(self):
        """Test cycles configuration validation ranges."""
        # Valid ranges
        CyclesConfig(max_cycles=1)
        CyclesConfig(max_cycles=100)
        CyclesConfig(max_reflect=0)
        CyclesConfig(max_reflect=10)
        
        # Invalid ranges
        with pytest.raises(ValueError):
            CyclesConfig(max_cycles=0)
        
        with pytest.raises(ValueError):
            CyclesConfig(max_cycles=101)
        
        with pytest.raises(ValueError):
            CyclesConfig(max_reflect=-1)
        
        with pytest.raises(ValueError):
            CyclesConfig(max_reflect=11)


class TestToolsConfig:
    """Test cases for ToolsConfig."""
    
    def test_valid_tools_config(self):
        """Test valid tools configuration."""
        config = ToolsConfig(
            enable=["code_generation", "static_analysis"],
            disabled=["autonomous_execution", "remote_actions"]
        )
        
        assert "code_generation" in config.enable
        assert "static_analysis" in config.enable
        assert "autonomous_execution" in config.disabled
        assert "remote_actions" in config.disabled
    
    def test_tools_config_defaults(self):
        """Test tools configuration defaults."""
        config = ToolsConfig()
        
        assert config.enable == []
        assert config.disabled == []