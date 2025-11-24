# orchestrator/nodes/base.py
"""
Base node architecture for consistent interfaces.
Provides abstract base class and common functionality for all processing nodes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from utils.exceptions import NodeError, ValidationError
from utils.logging_config import get_logger, log_with_context


@dataclass
class NodeResult:
    """
    Standard result structure for all nodes.
    
    Provides consistent output format across all processing nodes.
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Add node execution metadata."""
        if self.execution_time is None:
            self.execution_time = 0.0


@dataclass
class NodeConfig:
    """Configuration structure for nodes."""
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 1.0
    validate_input: bool = True
    validate_output: bool = True


class BaseNode(ABC):
    """
    Abstract base class for all processing nodes.
    
    Provides common functionality including:
    - Logging and monitoring
    - Input/output validation
    - Error handling
    - Execution timing
    - Configuration management
    """
    
    def __init__(self, config: Dict[str, Any], node_config: Optional[NodeConfig] = None):
        """
        Initialize base node.
        
        Args:
            config: Global configuration dictionary
            node_config: Node-specific configuration
        """
        self.config = config
        self.node_config = node_config or NodeConfig()
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._execution_count = 0
        self._total_execution_time = 0.0
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def execute(self, input_data: Any, **kwargs) -> NodeResult:
        """
        Execute the node with comprehensive error handling and monitoring.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional keyword arguments
            
        Returns:
            NodeResult with execution results
        """
        if not self.node_config.enabled:
            return NodeResult(
                success=False,
                error="Node is disabled",
                metadata={"node_name": self.__class__.__name__}
            )
        
        start_time = time.time()
        self._execution_count += 1
        
        try:
            # Input validation
            if self.node_config.validate_input:
                self._validate_input(input_data)
            
            log_with_context(
                self.logger, "info", 
                f"Starting execution #{self._execution_count}",
                node_name=self.__class__.__name__,
                execution_count=self._execution_count
            )
            
            # Execute with retry logic
            result_data = self._execute_with_retry(input_data, **kwargs)
            
            # Output validation
            if self.node_config.validate_output:
                self._validate_output(result_data)
            
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            
            result = NodeResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                metadata={
                    "node_name": self.__class__.__name__,
                    "execution_count": self._execution_count,
                    "average_execution_time": self._total_execution_time / self._execution_count
                }
            )
            
            log_with_context(
                self.logger, "info",
                f"Execution completed successfully",
                execution_time=execution_time,
                node_name=self.__class__.__name__
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Node execution failed: {str(e)}"
            
            log_with_context(
                self.logger, "error",
                error_msg,
                execution_time=execution_time,
                node_name=self.__class__.__name__,
                error_type=type(e).__name__
            )
            
            return NodeResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "node_name": self.__class__.__name__,
                    "execution_count": self._execution_count,
                    "error_type": type(e).__name__
                }
            )
    
    def _execute_with_retry(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute with retry logic.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional keyword arguments
            
        Returns:
            Execution result data
            
        Raises:
            NodeError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.node_config.retry_count + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt}/{self.node_config.retry_count}")
                    time.sleep(self.node_config.retry_delay)
                
                return self.run(input_data, **kwargs)
                
            except Exception as e:
                last_exception = e
                if attempt < self.node_config.retry_count:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                else:
                    self.logger.error(f"All retry attempts failed: {e}")
        
        raise NodeError(
            f"Node execution failed after {self.node_config.retry_count + 1} attempts",
            node_name=self.__class__.__name__,
            details={"last_error": str(last_exception)}
        )
    
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Core node processing logic.
        
        This method must be implemented by all concrete node classes.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            NodeError: If processing fails
        """
        pass
    
    def _validate_input(self, input_data: Any) -> None:
        """
        Validate input data.
        
        Override this method in concrete classes for specific validation.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValidationError: If input is invalid
        """
        if input_data is None:
            raise ValidationError("Input data cannot be None")
    
    def _validate_output(self, output_data: Dict[str, Any]) -> None:
        """
        Validate output data.
        
        Override this method in concrete classes for specific validation.
        
        Args:
            output_data: Output data to validate
            
        Raises:
            ValidationError: If output is invalid
        """
        if not isinstance(output_data, dict):
            raise ValidationError("Output data must be a dictionary")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get node execution statistics.
        
        Returns:
            Dictionary containing execution statistics
        """
        return {
            "node_name": self.__class__.__name__,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / self._execution_count 
                if self._execution_count > 0 else 0.0
            ),
            "enabled": self.node_config.enabled,
            "retry_count": self.node_config.retry_count
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_count = 0
        self._total_execution_time = 0.0
        self.logger.info("Node statistics reset")
    
    def enable(self) -> None:
        """Enable the node."""
        self.node_config.enabled = True
        self.logger.info("Node enabled")
    
    def disable(self) -> None:
        """Disable the node."""
        self.node_config.enabled = False
        self.logger.info("Node disabled")
    
    def is_enabled(self) -> bool:
        """Check if the node is enabled."""
        return self.node_config.enabled
    
    def cleanup(self) -> None:
        """
        Cleanup resources used by the node.
        
        Override this method in concrete classes for specific cleanup.
        """
        self.logger.info("Node cleanup completed")


class ProcessingNode(BaseNode):
    """
    Base class for nodes that process data.
    
    Provides additional functionality for data processing nodes.
    """
    
    def __init__(self, config: Dict[str, Any], node_config: Optional[NodeConfig] = None):
        super().__init__(config, node_config)
        self._processing_history = []
    
    def get_processing_history(self) -> list:
        """Get history of processing operations."""
        return self._processing_history.copy()
    
    def clear_history(self) -> None:
        """Clear processing history."""
        self._processing_history.clear()
        self.logger.info("Processing history cleared")
    
    def _record_processing(self, input_data: Any, output_data: Dict[str, Any]) -> None:
        """Record processing operation in history."""
        self._processing_history.append({
            "timestamp": datetime.now(),
            "input_size": len(str(input_data)) if input_data else 0,
            "output_size": len(str(output_data)) if output_data else 0,
            "success": True
        })
        
        # Keep only last 100 entries
        if len(self._processing_history) > 100:
            self._processing_history = self._processing_history[-100:]