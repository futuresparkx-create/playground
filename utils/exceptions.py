# utils/exceptions.py
"""
Custom exception hierarchy for the playground system.
Provides structured error handling with specific exception types.
"""


class PlaygroundException(Exception):
    """
    Base exception for all playground-related errors.
    
    All custom exceptions in the playground system should inherit from this.
    """
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(PlaygroundException):
    """
    Configuration-related errors.
    
    Raised when configuration files are missing, invalid, or contain
    incompatible settings.
    """
    pass


class ModelError(PlaygroundException):
    """
    Model-related errors.
    
    Raised when model loading, initialization, or inference fails.
    """
    pass


class NodeError(PlaygroundException):
    """
    Node processing errors.
    
    Raised when any processing node encounters an error during execution.
    """
    
    def __init__(self, message: str, node_name: str = None, details: dict = None):
        super().__init__(message, details)
        self.node_name = node_name


class SecurityError(PlaygroundException):
    """
    Security-related errors.
    
    Raised when security violations are detected, such as:
    - Dangerous input patterns
    - Path traversal attempts
    - Unauthorized operations
    """
    pass


class ValidationError(PlaygroundException):
    """
    Input validation errors.
    
    Raised when user input or data fails validation checks.
    """
    pass


class ResourceError(PlaygroundException):
    """
    Resource management errors.
    
    Raised when resource allocation, cleanup, or management fails.
    """
    pass


class MemoryError(PlaygroundException):
    """
    Memory system errors.
    
    Raised when memory storage, retrieval, or indexing operations fail.
    """
    pass


class OrchestrationError(PlaygroundException):
    """
    Orchestration and workflow errors.
    
    Raised when the improvement cycle or workflow coordination fails.
    """
    pass