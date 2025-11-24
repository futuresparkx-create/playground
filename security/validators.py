# security/validators.py
"""
Input validation and sanitization for security.
Provides comprehensive input validation to prevent security vulnerabilities.
"""

import re
import html
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse
import logging

from utils.exceptions import SecurityError, ValidationError

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Comprehensive input validator with security checks.
    
    Provides methods to validate and sanitize various types of user input
    to prevent security vulnerabilities.
    """
    
    # Dangerous patterns that could indicate code injection attempts
    DANGEROUS_PATTERNS = [
        r'__import__\s*\(',
        r'exec\s*\(',
        r'eval\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'subprocess\.',
        r'os\.',
        r'sys\.',
        r'importlib\.',
        r'__builtins__',
        r'__globals__',
        r'__locals__',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'hasattr\s*\(',
        r'delattr\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'globals\s*\(',
        r'locals\s*\(',
    ]
    
    # File extension whitelist for safe operations
    SAFE_EXTENSIONS = {'.py', '.js', '.ts', '.html', '.css', '.json', '.yaml', '.yml', '.md', '.txt'}
    
    # Maximum input lengths
    MAX_TASK_LENGTH = 10000
    MAX_CODE_LENGTH = 50000
    MAX_PATH_LENGTH = 500
    
    @classmethod
    def sanitize_task_input(cls, task: str) -> str:
        """
        Sanitize user input for coding tasks.
        
        Args:
            task: User input task description
            
        Returns:
            Sanitized task string
            
        Raises:
            SecurityError: If dangerous patterns are detected
            ValidationError: If input is invalid
        """
        if not isinstance(task, str):
            raise ValidationError("Task input must be a string")
        
        if not task.strip():
            raise ValidationError("Task input cannot be empty")
        
        if len(task) > cls.MAX_TASK_LENGTH:
            raise ValidationError(f"Task input too long (max {cls.MAX_TASK_LENGTH} characters)")
        
        # Check for dangerous patterns
        cls._check_dangerous_patterns(task, "task input")
        
        # HTML escape to prevent XSS
        sanitized = html.escape(task)
        
        # Remove null bytes and other control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        logger.debug(f"Task input sanitized: {len(task)} -> {len(sanitized)} characters")
        
        return sanitized.strip()
    
    @classmethod
    def sanitize_code_input(cls, code: str, language: str = "python") -> str:
        """
        Sanitize code input for analysis.
        
        Args:
            code: Code content to sanitize
            language: Programming language (for context)
            
        Returns:
            Sanitized code string
            
        Raises:
            SecurityError: If dangerous patterns are detected
            ValidationError: If input is invalid
        """
        if not isinstance(code, str):
            raise ValidationError("Code input must be a string")
        
        if len(code) > cls.MAX_CODE_LENGTH:
            raise ValidationError(f"Code input too long (max {cls.MAX_CODE_LENGTH} characters)")
        
        # For Python code, check for dangerous patterns
        if language.lower() == "python":
            cls._check_dangerous_patterns(code, "Python code")
        
        # Remove null bytes
        sanitized = re.sub(r'\x00', '', code)
        
        logger.debug(f"Code input sanitized for {language}: {len(code)} -> {len(sanitized)} characters")
        
        return sanitized
    
    @classmethod
    def validate_file_path(cls, path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
        """
        Validate and sanitize file paths to prevent path traversal attacks.
        
        Args:
            path: File path to validate
            base_dir: Base directory to restrict access to
            
        Returns:
            Validated and resolved Path object
            
        Raises:
            SecurityError: If path traversal is detected
            ValidationError: If path is invalid
        """
        if isinstance(path, str):
            if len(path) > cls.MAX_PATH_LENGTH:
                raise ValidationError(f"Path too long (max {cls.MAX_PATH_LENGTH} characters)")
            path = Path(path)
        
        if not isinstance(path, Path):
            raise ValidationError("Path must be a string or Path object")
        
        # Resolve the path to handle .. and . components
        try:
            resolved_path = path.resolve()
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid path: {e}")
        
        # Check for path traversal if base directory is specified
        if base_dir:
            base_dir = base_dir.resolve()
            try:
                resolved_path.relative_to(base_dir)
            except ValueError:
                raise SecurityError(f"Path traversal attempt detected: {path}")
        
        # Check file extension
        if resolved_path.suffix and resolved_path.suffix.lower() not in cls.SAFE_EXTENSIONS:
            logger.warning(f"Potentially unsafe file extension: {resolved_path.suffix}")
        
        logger.debug(f"Path validated: {path} -> {resolved_path}")
        
        return resolved_path
    
    @classmethod
    def validate_url(cls, url: str) -> str:
        """
        Validate URL input for safety.
        
        Args:
            url: URL to validate
            
        Returns:
            Validated URL string
            
        Raises:
            ValidationError: If URL is invalid or unsafe
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")
        
        if len(url) > 2000:  # Reasonable URL length limit
            raise ValidationError("URL too long")
        
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")
        
        # Only allow safe schemes
        safe_schemes = {'http', 'https', 'ftp', 'ftps'}
        if parsed.scheme.lower() not in safe_schemes:
            raise ValidationError(f"Unsafe URL scheme: {parsed.scheme}")
        
        # Prevent localhost/internal network access in production
        if parsed.hostname:
            if parsed.hostname.lower() in ['localhost', '127.0.0.1', '::1']:
                logger.warning(f"Localhost URL detected: {url}")
        
        return url
    
    @classmethod
    def _check_dangerous_patterns(cls, text: str, context: str) -> None:
        """
        Check text for dangerous patterns that could indicate code injection.
        
        Args:
            text: Text to check
            context: Context description for error messages
            
        Raises:
            SecurityError: If dangerous patterns are found
        """
        text_lower = text.lower()
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected in {context}: {pattern}")
                raise SecurityError(f"Dangerous pattern detected in {context}: {pattern}")
        
        # Check for suspicious character sequences
        if re.search(r'[^\x20-\x7E\n\r\t]', text):
            # Contains non-printable characters (excluding common whitespace)
            logger.warning(f"Non-printable characters detected in {context}")
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        """
        Validate model name for safety.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            Validated model name
            
        Raises:
            ValidationError: If model name is invalid
        """
        if not isinstance(model_name, str):
            raise ValidationError("Model name must be a string")
        
        if not model_name.strip():
            raise ValidationError("Model name cannot be empty")
        
        if len(model_name) > 100:
            raise ValidationError("Model name too long")
        
        # Allow only alphanumeric, hyphens, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
            raise ValidationError("Model name contains invalid characters")
        
        return model_name.strip()
    
    @classmethod
    def validate_config_value(cls, value: Union[str, int, float, bool], value_type: str) -> Union[str, int, float, bool]:
        """
        Validate configuration values.
        
        Args:
            value: Configuration value to validate
            value_type: Expected type of the value
            
        Returns:
            Validated configuration value
            
        Raises:
            ValidationError: If value is invalid
        """
        if value_type == "string":
            if not isinstance(value, str):
                raise ValidationError(f"Expected string, got {type(value).__name__}")
            if len(value) > 1000:
                raise ValidationError("String value too long")
            return value
        
        elif value_type == "integer":
            if not isinstance(value, int):
                raise ValidationError(f"Expected integer, got {type(value).__name__}")
            if not -1000000 <= value <= 1000000:
                raise ValidationError("Integer value out of range")
            return value
        
        elif value_type == "float":
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Expected float, got {type(value).__name__}")
            if not -1000000.0 <= value <= 1000000.0:
                raise ValidationError("Float value out of range")
            return float(value)
        
        elif value_type == "boolean":
            if not isinstance(value, bool):
                raise ValidationError(f"Expected boolean, got {type(value).__name__}")
            return value
        
        else:
            raise ValidationError(f"Unknown value type: {value_type}")


class RateLimiter:
    """
    Simple rate limiter for API calls and operations.
    
    Helps prevent abuse and resource exhaustion.
    """
    
    def __init__(self, max_calls: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def is_allowed(self) -> bool:
        """
        Check if a new call is allowed.
        
        Returns:
            True if call is allowed, False otherwise
        """
        import time
        
        current_time = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if current_time - call_time < self.time_window]
        
        # Check if we're under the limit
        if len(self.calls) < self.max_calls:
            self.calls.append(current_time)
            return True
        
        return False
    
    def get_remaining_calls(self) -> int:
        """Get number of remaining calls in current window."""
        import time
        
        current_time = time.time()
        self.calls = [call_time for call_time in self.calls 
                     if current_time - call_time < self.time_window]
        
        return max(0, self.max_calls - len(self.calls))