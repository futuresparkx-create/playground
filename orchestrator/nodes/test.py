# orchestrator/nodes/test.py
"""
Static analysis node for code quality and safety checks.
Performs comprehensive static analysis without executing code.
"""

import subprocess
import tempfile
import uuid
import json
import re
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union

from .base import ProcessingNode, NodeConfig
from security.validators import InputValidator
from utils.exceptions import NodeError, ValidationError, ResourceError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class TestNode(ProcessingNode):
    """
    Static analysis node for code quality and safety checks.
    
    Performs comprehensive static analysis using multiple tools:
    - Python: ruff (linting) + pyright (type checking)
    - JavaScript/TypeScript: eslint
    - General: custom security pattern detection
    
    Never executes code - static analysis only for safety.
    """
    
    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "python": [
            r'\bdef\s+\w+\s*\(',
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import',
            r'\bclass\s+\w+\s*\(',
            r'^\s*#.*python',
        ],
        "javascript": [
            r'\bfunction\s+\w+\s*\(',
            r'\bconst\s+\w+\s*=',
            r'\blet\s+\w+\s*=',
            r'\bvar\s+\w+\s*=',
            r'^\s*//.*javascript',
        ],
        "typescript": [
            r'\binterface\s+\w+',
            r'\btype\s+\w+\s*=',
            r':\s*\w+\s*=',
            r'\bexport\s+\w+',
            r'^\s*//.*typescript',
        ]
    }
    
    # Security patterns to detect
    SECURITY_PATTERNS = {
        "dangerous_imports": [
            r'\bos\.',
            r'\bsys\.',
            r'\bsubprocess\.',
            r'\b__import__\s*\(',
            r'\beval\s*\(',
            r'\bexec\s*\(',
        ],
        "file_operations": [
            r'\bopen\s*\(',
            r'\bfile\s*\(',
            r'\.read\s*\(',
            r'\.write\s*\(',
        ],
        "network_operations": [
            r'\brequests\.',
            r'\burllib\.',
            r'\bsocket\.',
            r'\bhttplib\.',
        ]
    }
    
    def __init__(self, config: Dict[str, Any], node_config: NodeConfig = None):
        """
        Initialize the test node.
        
        Args:
            config: Global configuration dictionary
            node_config: Node-specific configuration
        """
        super().__init__(config, node_config)
        
        # Tool availability check
        self.available_tools = self._check_tool_availability()
        
        logger.info(f"TestNode initialized with tools: {list(self.available_tools.keys())}")
    
    def run(self, code: str, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform static analysis on the provided code.
        
        Args:
            code: Code to analyze
            language: Optional language hint
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            NodeError: If analysis fails
            ValidationError: If input validation fails
        """
        try:
            # Validate and sanitize input
            sanitized_code = InputValidator.sanitize_code_input(code, language or "unknown")
            
            # Detect language if not provided
            detected_language = language or self._detect_language(sanitized_code)
            
            if not detected_language:
                return self._create_result("unknown_language", [], [], {
                    "message": "Could not detect programming language"
                })
            
            logger.info(f"Analyzing {detected_language} code ({len(sanitized_code)} characters)")
            
            # Perform security analysis first
            security_issues = self._analyze_security_patterns(sanitized_code, detected_language)
            
            # Perform language-specific analysis
            if detected_language == "python":
                analysis_result = self._analyze_python(sanitized_code)
            elif detected_language in ["javascript", "typescript"]:
                analysis_result = self._analyze_javascript(sanitized_code, detected_language)
            else:
                analysis_result = self._create_result(
                    f"unsupported_language_{detected_language}", [], [], {
                        "message": f"Static analysis not supported for {detected_language}"
                    }
                )
            
            # Merge security issues with analysis results
            if security_issues:
                analysis_result["errors"].extend(security_issues)
            
            # Add metadata
            analysis_result["metadata"].update({
                "language": detected_language,
                "code_length": len(sanitized_code),
                "line_count": len(sanitized_code.splitlines()),
                "available_tools": self.available_tools
            })
            
            # Record processing for history
            self._record_processing(code, analysis_result)
            
            logger.info(f"Static analysis completed: {len(analysis_result['errors'])} errors, "
                       f"{len(analysis_result['warnings'])} warnings")
            
            return analysis_result
            
        except ValidationError as e:
            raise NodeError(f"Input validation failed: {e}", node_name="TestNode")
        except Exception as e:
            logger.error(f"Static analysis failed: {e}")
            raise NodeError(f"Static analysis failed: {e}", node_name="TestNode")
    
    def _detect_language(self, code: str) -> Optional[str]:
        """
        Detect programming language from code content.
        
        Args:
            code: Code to analyze
            
        Returns:
            Detected language or None
        """
        code_lower = code.lower()
        
        # Score each language based on pattern matches
        language_scores = {}
        
        for language, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
                score += matches
            
            if score > 0:
                language_scores[language] = score
        
        if not language_scores:
            return None
        
        # Return language with highest score
        detected = max(language_scores, key=language_scores.get)
        logger.debug(f"Language detection scores: {language_scores}, selected: {detected}")
        
        return detected
    
    def _analyze_security_patterns(self, code: str, language: str) -> List[Dict[str, Any]]:
        """
        Analyze code for security patterns.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            List of security issues found
        """
        security_issues = []
        
        for category, patterns in self.SECURITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    security_issues.append({
                        "type": "security_warning",
                        "category": category,
                        "message": f"Potentially dangerous pattern detected: {match.group()}",
                        "line": line_num,
                        "column": match.start() - code.rfind('\n', 0, match.start()),
                        "severity": "warning"
                    })
        
        return security_issues
    
    @contextmanager
    def _temp_file(self, content: str, suffix: str):
        """
        Context manager for temporary files with automatic cleanup.
        
        Args:
            content: File content
            suffix: File extension
            
        Yields:
            Path to temporary file
        """
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=suffix, 
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(content)
                temp_file = Path(f.name)
            
            yield temp_file
            
        except Exception as e:
            logger.error(f"Temporary file operation failed: {e}")
            raise ResourceError(f"Temporary file operation failed: {e}")
        finally:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file {temp_file}: {e}")
    
    def _analyze_python(self, code: str) -> Dict[str, Any]:
        """
        Analyze Python code using ruff and pyright.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Analysis results
        """
        errors = []
        warnings = []
        
        with self._temp_file(code, '.py') as temp_path:
            # Run ruff linter
            if self.available_tools.get("ruff"):
                try:
                    ruff_result = subprocess.run(
                        ["ruff", "check", str(temp_path), "--format=json"],
                        capture_output=True, 
                        text=True, 
                        timeout=30
                    )
                    
                    ruff_issues = self._parse_tool_output(ruff_result.stdout, "ruff")
                    errors.extend(ruff_issues)
                    
                except subprocess.TimeoutExpired:
                    logger.warning("Ruff analysis timed out")
                    warnings.append({
                        "type": "tool_timeout",
                        "message": "Ruff analysis timed out",
                        "tool": "ruff"
                    })
                except Exception as e:
                    logger.warning(f"Ruff analysis failed: {e}")
                    warnings.append({
                        "type": "tool_error",
                        "message": f"Ruff analysis failed: {e}",
                        "tool": "ruff"
                    })
            
            # Run pyright type checker
            if self.available_tools.get("pyright"):
                try:
                    pyright_result = subprocess.run(
                        ["pyright", str(temp_path), "--outputjson"],
                        capture_output=True, 
                        text=True, 
                        timeout=30
                    )
                    
                    pyright_issues = self._parse_tool_output(pyright_result.stdout, "pyright")
                    errors.extend(pyright_issues)
                    
                except subprocess.TimeoutExpired:
                    logger.warning("Pyright analysis timed out")
                    warnings.append({
                        "type": "tool_timeout",
                        "message": "Pyright analysis timed out",
                        "tool": "pyright"
                    })
                except Exception as e:
                    logger.warning(f"Pyright analysis failed: {e}")
                    warnings.append({
                        "type": "tool_error",
                        "message": f"Pyright analysis failed: {e}",
                        "tool": "pyright"
                    })
        
        return self._create_result("python_analysis", errors, warnings)
    
    def _analyze_javascript(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze JavaScript/TypeScript code using eslint.
        
        Args:
            code: JavaScript/TypeScript code to analyze
            language: Specific language (javascript or typescript)
            
        Returns:
            Analysis results
        """
        errors = []
        warnings = []
        
        file_extension = '.ts' if language == 'typescript' else '.js'
        
        with self._temp_file(code, file_extension) as temp_path:
            # Run eslint
            if self.available_tools.get("eslint"):
                try:
                    eslint_result = subprocess.run(
                        ["eslint", str(temp_path), "-f", "json"],
                        capture_output=True, 
                        text=True, 
                        timeout=30
                    )
                    
                    eslint_issues = self._parse_tool_output(eslint_result.stdout, "eslint")
                    errors.extend(eslint_issues)
                    
                except subprocess.TimeoutExpired:
                    logger.warning("ESLint analysis timed out")
                    warnings.append({
                        "type": "tool_timeout",
                        "message": "ESLint analysis timed out",
                        "tool": "eslint"
                    })
                except Exception as e:
                    logger.warning(f"ESLint analysis failed: {e}")
                    warnings.append({
                        "type": "tool_error",
                        "message": f"ESLint analysis failed: {e}",
                        "tool": "eslint"
                    })
        
        return self._create_result(f"{language}_analysis", errors, warnings)
    
    def _parse_tool_output(self, output: str, tool: str) -> List[Dict[str, Any]]:
        """
        Parse tool output into standardized format.
        
        Args:
            output: Tool output string
            tool: Tool name
            
        Returns:
            List of parsed issues
        """
        if not output.strip():
            return []
        
        try:
            parsed_output = json.loads(output)
            
            # Handle different tool output formats
            if tool == "ruff":
                return self._parse_ruff_output(parsed_output)
            elif tool == "pyright":
                return self._parse_pyright_output(parsed_output)
            elif tool == "eslint":
                return self._parse_eslint_output(parsed_output)
            else:
                return []
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {tool} output as JSON")
            return [{
                "type": "parse_error",
                "message": f"Failed to parse {tool} output",
                "tool": tool
            }]
        except Exception as e:
            logger.warning(f"Error parsing {tool} output: {e}")
            return []
    
    def _parse_ruff_output(self, output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse ruff output format."""
        issues = []
        for item in output:
            issues.append({
                "type": "lint_error",
                "message": item.get("message", "Unknown error"),
                "line": item.get("location", {}).get("row", 0),
                "column": item.get("location", {}).get("column", 0),
                "rule": item.get("code", "unknown"),
                "severity": "error",
                "tool": "ruff"
            })
        return issues
    
    def _parse_pyright_output(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse pyright output format."""
        issues = []
        diagnostics = output.get("generalDiagnostics", [])
        
        for diagnostic in diagnostics:
            issues.append({
                "type": "type_error",
                "message": diagnostic.get("message", "Unknown error"),
                "line": diagnostic.get("range", {}).get("start", {}).get("line", 0) + 1,
                "column": diagnostic.get("range", {}).get("start", {}).get("character", 0),
                "severity": diagnostic.get("severity", "error"),
                "tool": "pyright"
            })
        
        return issues
    
    def _parse_eslint_output(self, output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse eslint output format."""
        issues = []
        
        for file_result in output:
            for message in file_result.get("messages", []):
                issues.append({
                    "type": "lint_error",
                    "message": message.get("message", "Unknown error"),
                    "line": message.get("line", 0),
                    "column": message.get("column", 0),
                    "rule": message.get("ruleId", "unknown"),
                    "severity": message.get("severity", 1) == 2 and "error" or "warning",
                    "tool": "eslint"
                })
        
        return issues
    
    def _create_result(self, analysis_type: str, errors: List[Dict[str, Any]], 
                      warnings: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized result dictionary.
        
        Args:
            analysis_type: Type of analysis performed
            errors: List of errors found
            warnings: List of warnings found
            metadata: Additional metadata
            
        Returns:
            Standardized result dictionary
        """
        return {
            "static_result": analysis_type,
            "errors": errors,
            "warnings": warnings,
            "metadata": metadata or {}
        }
    
    def _check_tool_availability(self) -> Dict[str, bool]:
        """
        Check availability of static analysis tools.
        
        Returns:
            Dictionary mapping tool names to availability
        """
        tools = {}
        
        # Check ruff
        try:
            subprocess.run(["ruff", "--version"], capture_output=True, timeout=5)
            tools["ruff"] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["ruff"] = False
        
        # Check pyright
        try:
            subprocess.run(["pyright", "--version"], capture_output=True, timeout=5)
            tools["pyright"] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["pyright"] = False
        
        # Check eslint
        try:
            subprocess.run(["eslint", "--version"], capture_output=True, timeout=5)
            tools["eslint"] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["eslint"] = False
        
        logger.info(f"Tool availability check: {tools}")
        return tools
    
    def _validate_input(self, input_data: Any) -> None:
        """
        Validate input data for static analysis.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValidationError: If input is invalid
        """
        super()._validate_input(input_data)
        
        if not isinstance(input_data, str):
            raise ValidationError("Input must be a string containing code")
        
        if not input_data.strip():
            raise ValidationError("Code input cannot be empty")
        
        if len(input_data) > 100000:  # 100KB limit
            raise ValidationError("Code input too large (max 100KB)")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get analysis-specific statistics.
        
        Returns:
            Dictionary with analysis statistics
        """
        base_stats = self.get_stats()
        
        # Add analysis-specific stats
        base_stats.update({
            "available_tools": self.available_tools,
            "supported_languages": list(self.LANGUAGE_PATTERNS.keys()),
            "security_patterns": len(sum(self.SECURITY_PATTERNS.values(), [])),
            "processing_history_size": len(self.get_processing_history())
        })
        
        return base_stats