# models/model_factory.py
"""
Model factory pattern for managing expensive model instances.
Provides singleton pattern and lifecycle management for AI models.
"""

from typing import Dict, Any, Optional, Type
import logging
import threading
from dataclasses import dataclass
from datetime import datetime

from utils.exceptions import ModelError
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a model instance."""
    model_name: str
    engine: str
    created_at: datetime
    last_used: datetime
    usage_count: int = 0
    memory_usage: Optional[float] = None


class ModelFactory:
    """
    Factory for creating and managing model instances.
    
    Implements singleton pattern for expensive model instances to avoid
    redundant loading and improve resource utilization.
    """
    
    _instances: Dict[str, Any] = {}
    _model_info: Dict[str, ModelInfo] = {}
    _lock = threading.Lock()
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> Any:
        """
        Create or return cached model instance.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Model instance (cached or newly created)
            
        Raises:
            ModelError: If model creation fails
        """
        cache_key = cls._generate_cache_key(config)
        
        with cls._lock:
            if cache_key in cls._instances:
                # Update usage statistics
                cls._model_info[cache_key].last_used = datetime.now()
                cls._model_info[cache_key].usage_count += 1
                
                logger.debug(f"Returning cached model instance: {cache_key}")
                return cls._instances[cache_key]
            
            # Create new model instance
            try:
                logger.info(f"Creating new model instance: {cache_key}")
                model_instance = cls._create_model_instance(config)
                
                # Cache the instance
                cls._instances[cache_key] = model_instance
                cls._model_info[cache_key] = ModelInfo(
                    model_name=config.get('name', 'unknown'),
                    engine=config.get('engine', 'unknown'),
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    usage_count=1
                )
                
                logger.info(f"Model instance created and cached: {cache_key}")
                return model_instance
                
            except Exception as e:
                logger.error(f"Failed to create model instance: {e}")
                raise ModelError(f"Model creation failed: {e}")
    
    @classmethod
    def _create_model_instance(cls, config: Dict[str, Any]) -> Any:
        """
        Create a new model instance based on configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            New model instance
            
        Raises:
            ModelError: If model creation fails
        """
        try:
            # Import here to avoid circular imports
            from models.load_model import ModelWrapper
            
            return ModelWrapper(config)
            
        except ImportError as e:
            raise ModelError(f"Failed to import ModelWrapper: {e}")
        except Exception as e:
            raise ModelError(f"Failed to create ModelWrapper: {e}")
    
    @classmethod
    def _generate_cache_key(cls, config: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for the model configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Unique cache key string
        """
        model_name = config.get('name', 'default')
        engine = config.get('engine', 'default')
        cache_dir = config.get('cache_dir', 'default')
        
        # Include relevant parameters that affect model identity
        key_parts = [
            f"name:{model_name}",
            f"engine:{engine}",
            f"cache:{cache_dir}"
        ]
        
        # Add other relevant config parameters
        for param in ['max_tokens', 'temperature', 'top_p']:
            if param in config:
                key_parts.append(f"{param}:{config[param]}")
        
        return "_".join(key_parts)
    
    @classmethod
    def get_model_info(cls, cache_key: Optional[str] = None) -> Dict[str, ModelInfo]:
        """
        Get information about cached models.
        
        Args:
            cache_key: Specific model cache key (optional)
            
        Returns:
            Dictionary of model information
        """
        with cls._lock:
            if cache_key:
                return {cache_key: cls._model_info.get(cache_key)}
            return cls._model_info.copy()
    
    @classmethod
    def get_cached_models(cls) -> Dict[str, Any]:
        """
        Get all cached model instances.
        
        Returns:
            Dictionary of cached model instances
        """
        with cls._lock:
            return cls._instances.copy()
    
    @classmethod
    def remove_model(cls, cache_key: str) -> bool:
        """
        Remove a specific model from cache.
        
        Args:
            cache_key: Cache key of model to remove
            
        Returns:
            True if model was removed, False if not found
        """
        with cls._lock:
            if cache_key in cls._instances:
                # Cleanup model if it has a cleanup method
                model = cls._instances[cache_key]
                if hasattr(model, 'cleanup'):
                    try:
                        model.cleanup()
                        logger.info(f"Model cleanup completed: {cache_key}")
                    except Exception as e:
                        logger.warning(f"Model cleanup failed: {e}")
                
                # Remove from cache
                del cls._instances[cache_key]
                del cls._model_info[cache_key]
                
                logger.info(f"Model removed from cache: {cache_key}")
                return True
            
            return False
    
    @classmethod
    def cleanup_all(cls) -> None:
        """
        Cleanup all cached model instances.
        
        This method should be called when shutting down the application
        to properly release resources.
        """
        with cls._lock:
            logger.info(f"Cleaning up {len(cls._instances)} cached models")
            
            for cache_key, model in cls._instances.items():
                try:
                    if hasattr(model, 'cleanup'):
                        model.cleanup()
                        logger.debug(f"Model cleanup completed: {cache_key}")
                except Exception as e:
                    logger.warning(f"Model cleanup failed for {cache_key}: {e}")
            
            cls._instances.clear()
            cls._model_info.clear()
            
            logger.info("All models cleaned up")
    
    @classmethod
    def cleanup_unused(cls, max_age_hours: float = 24.0) -> int:
        """
        Cleanup models that haven't been used recently.
        
        Args:
            max_age_hours: Maximum age in hours for unused models
            
        Returns:
            Number of models cleaned up
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        models_to_remove = []
        
        with cls._lock:
            for cache_key, info in cls._model_info.items():
                if info.last_used < cutoff_time:
                    models_to_remove.append(cache_key)
        
        # Remove unused models
        cleaned_count = 0
        for cache_key in models_to_remove:
            if cls.remove_model(cache_key):
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} unused models")
        
        return cleaned_count
    
    @classmethod
    def get_memory_usage(cls) -> Dict[str, Any]:
        """
        Get memory usage statistics for cached models.
        
        Returns:
            Dictionary with memory usage information
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "total_models": len(cls._instances),
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "process_memory_percent": process.memory_percent(),
            "models": {
                cache_key: {
                    "usage_count": info.usage_count,
                    "age_hours": (datetime.now() - info.created_at).total_seconds() / 3600,
                    "last_used_hours": (datetime.now() - info.last_used).total_seconds() / 3600
                }
                for cache_key, info in cls._model_info.items()
            }
        }
    
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """
        Perform health check on all cached models.
        
        Returns:
            Health check results
        """
        results = {
            "total_models": len(cls._instances),
            "healthy_models": 0,
            "unhealthy_models": 0,
            "model_status": {}
        }
        
        with cls._lock:
            for cache_key, model in cls._instances.items():
                try:
                    # Try to perform a simple operation to check health
                    if hasattr(model, 'health_check'):
                        health_status = model.health_check()
                    else:
                        # Basic check - just verify the model object exists
                        health_status = model is not None
                    
                    if health_status:
                        results["healthy_models"] += 1
                        results["model_status"][cache_key] = "healthy"
                    else:
                        results["unhealthy_models"] += 1
                        results["model_status"][cache_key] = "unhealthy"
                        
                except Exception as e:
                    results["unhealthy_models"] += 1
                    results["model_status"][cache_key] = f"error: {str(e)}"
                    logger.warning(f"Health check failed for model {cache_key}: {e}")
        
        return results