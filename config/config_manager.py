# config/config_manager.py
"""
Centralized configuration management with Pydantic validation.
Provides type-safe configuration loading and validation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, ValidationError, Field
import logging

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration with validation."""
    name: str = Field(..., description="Model name")
    engine: str = Field(..., description="Model engine")
    max_tokens: int = Field(8192, ge=1, le=32768, description="Maximum tokens")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature for generation")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p for generation")


class CyclesConfig(BaseModel):
    """Cycles configuration with validation."""
    max_cycles: int = Field(10, ge=1, le=100, description="Maximum improvement cycles")
    max_reflect: int = Field(3, ge=0, le=10, description="Maximum reflection iterations")
    require_human_approval: bool = Field(True, description="Require human approval for cycles")


class ToolsConfig(BaseModel):
    """Tools configuration with validation."""
    enable: list[str] = Field(default_factory=list, description="Enabled tools")
    disabled: list[str] = Field(default_factory=list, description="Disabled tools")


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


class ConfigManager:
    """
    Centralized configuration manager with caching and validation.
    
    Provides type-safe configuration loading with Pydantic validation
    and caching for performance.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        self._config_cache: Dict[str, Any] = {}
        self._ensure_config_dir()
    
    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        if not self.config_dir.exists():
            raise ConfigurationError(f"Configuration directory not found: {self.config_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate all configuration files.
        
        Returns:
            Dict containing validated configuration objects
            
        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        if not self._config_cache:
            try:
                self._config_cache = {
                    "model": self._load_and_validate(ModelConfig, "model.yaml"),
                    "cycles": self._load_and_validate(CyclesConfig, "cycles.yaml"),
                    "tools": self._load_and_validate(ToolsConfig, "tools.yaml")
                }
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise ConfigurationError(f"Configuration loading failed: {e}")
        
        return self._config_cache
    
    def _load_and_validate(self, model_class: type[BaseModel], filename: str) -> BaseModel:
        """
        Load and validate a specific configuration file.
        
        Args:
            model_class: Pydantic model class for validation
            filename: Configuration file name
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigurationError: If file loading or validation fails
        """
        file_path = self.config_dir / filename
        
        try:
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {file_path}, using defaults")
                return model_class()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            # Extract the nested configuration if it exists
            # e.g., model.yaml has 'model:' key, cycles.yaml has 'cycles:' key
            config_key = filename.split('.')[0]
            if config_key in data:
                data = data[config_key]
            
            return model_class(**data)
            
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}, using defaults")
            return model_class()
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {filename}: {e}")
        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration in {filename}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Unexpected error loading {filename}: {e}")
    
    def get_model_config(self) -> ModelConfig:
        """Get validated model configuration."""
        return self.load_config()["model"]
    
    def get_cycles_config(self) -> CyclesConfig:
        """Get validated cycles configuration."""
        return self.load_config()["cycles"]
    
    def get_tools_config(self) -> ToolsConfig:
        """Get validated tools configuration."""
        return self.load_config()["tools"]
    
    def reload_config(self) -> None:
        """Clear cache and reload configuration."""
        self._config_cache.clear()
        logger.info("Configuration cache cleared, will reload on next access")
    
    def validate_config_files(self) -> bool:
        """
        Validate all configuration files without caching.
        
        Returns:
            True if all configurations are valid
            
        Raises:
            ConfigurationError: If any configuration is invalid
        """
        try:
            # Temporarily clear cache to force reload
            original_cache = self._config_cache.copy()
            self._config_cache.clear()
            
            # Load and validate all configs
            self.load_config()
            
            # Restore cache
            self._config_cache = original_cache
            
            return True
        except ConfigurationError:
            # Restore cache on error
            self._config_cache = original_cache
            raise