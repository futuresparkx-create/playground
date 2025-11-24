# utils/logging_config.py
"""
Structured logging configuration for the playground system.
Provides consistent logging setup with appropriate formatters and handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON for better parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class SecurityFilter(logging.Filter):
    """
    Security filter to prevent sensitive information from being logged.
    
    Filters out potentially sensitive data from log messages.
    """
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'key', 'secret', 'auth',
        'credential', 'api_key', 'access_token'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out records containing sensitive information."""
        message = record.getMessage().lower()
        
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                # Replace the message with a sanitized version
                record.msg = "[REDACTED - Sensitive information filtered]"
                record.args = ()
                break
        
        return True


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    enable_security_filter: bool = True
) -> None:
    """
    Configure structured logging for the playground system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Whether to use JSON formatting
        enable_security_filter: Whether to enable security filtering
    """
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add security filter if enabled
    if enable_security_filter:
        security_filter = SecurityFilter()
        console_handler.addFilter(security_filter)
    
    handlers = [console_handler]
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        if enable_security_filter:
            file_handler.addFilter(security_filter)
        
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: str, message: str, **context) -> None:
    """
    Log a message with additional context.
    
    Args:
        logger: Logger instance
        level: Log level (info, warning, error, etc.)
        message: Log message
        **context: Additional context to include in the log
    """
    # Create a log record with extra context
    log_method = getattr(logger, level.lower())
    
    # Add context as extra fields for JSON formatter
    extra = {"extra_fields": context} if context else {}
    
    log_method(message, extra=extra)