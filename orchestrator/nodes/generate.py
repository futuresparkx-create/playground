# orchestrator/nodes/generate.py
"""
Code generation node using SGLang with TensorRT-LLM.
Generates code solutions based on user tasks with structured output.
"""

from typing import Dict, Any
import json

from .base import ProcessingNode, NodeConfig
from models.model_factory import ModelFactory
from security.validators import InputValidator
from utils.exceptions import NodeError, ValidationError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class GenerateNode(ProcessingNode):
    """
    Node for generating code solutions using AI models.
    
    Uses SGLang with TensorRT-LLM backend for efficient code generation
    with structured output validation.
    """
    
    def __init__(self, config: Dict[str, Any], node_config: NodeConfig = None):
        """
        Initialize the generation node.
        
        Args:
            config: Global configuration dictionary
            node_config: Node-specific configuration
        """
        super().__init__(config, node_config)
        
        # Get model configuration
        self.model_config = config.get("model", {})
        
        # Initialize model through factory
        self.model = ModelFactory.create_model(self.model_config)
        
        # Generation schema for structured output
        self.generation_schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Explanation of the solution approach"
                },
                "code": {
                    "type": "string", 
                    "description": "Generated code solution"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language used"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the solution"
                }
            },
            "required": ["answer", "code"]
        }
        
        logger.info("GenerateNode initialized with model factory")
    
    def run(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Generate code solution for the given task.
        
        Args:
            task: User task description
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generation results
            
        Raises:
            NodeError: If generation fails
            ValidationError: If input validation fails
        """
        try:
            # Validate and sanitize input
            sanitized_task = InputValidator.sanitize_task_input(task)
            
            # Prepare generation parameters
            generation_params = self._prepare_generation_params(kwargs)
            
            # Generate solution
            logger.info(f"Generating solution for task: {sanitized_task[:100]}...")
            
            result = self.model.generate(
                sanitized_task, 
                self.generation_schema,
                **generation_params
            )
            
            # Process and validate output
            processed_result = self._process_generation_result(result, sanitized_task)
            
            # Record processing for history
            self._record_processing(sanitized_task, processed_result)
            
            logger.info("Code generation completed successfully")
            
            return processed_result
            
        except ValidationError as e:
            raise NodeError(f"Input validation failed: {e}", node_name="GenerateNode")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise NodeError(f"Code generation failed: {e}", node_name="GenerateNode")
    
    def _prepare_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare generation parameters from config and kwargs.
        
        Args:
            kwargs: Additional parameters
            
        Returns:
            Dictionary of generation parameters
        """
        params = {
            "max_tokens": self.model_config.get("max_tokens", 8192),
            "temperature": self.model_config.get("temperature", 0.1),
            "top_p": self.model_config.get("top_p", 0.95)
        }
        
        # Override with kwargs if provided
        for key in ["max_tokens", "temperature", "top_p"]:
            if key in kwargs:
                params[key] = kwargs[key]
        
        return params
    
    def _process_generation_result(self, result: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """
        Process and validate generation result.
        
        Args:
            result: Raw generation result from model
            original_task: Original task for context
            
        Returns:
            Processed result dictionary
            
        Raises:
            NodeError: If result processing fails
        """
        try:
            # Extract output and validate schema
            output = result.get("output", {})
            schema_valid = result.get("schema_valid", False)
            
            # Parse output if it's a string
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a structured response
                    output = {
                        "answer": "Generated response (parsing failed)",
                        "code": output,
                        "language": "unknown"
                    }
                    schema_valid = False
            
            # Validate required fields
            if not isinstance(output, dict):
                raise NodeError("Generated output is not a dictionary")
            
            if "answer" not in output or "code" not in output:
                raise NodeError("Generated output missing required fields")
            
            # Sanitize code output
            if output.get("code"):
                language = output.get("language", "python")
                output["code"] = InputValidator.sanitize_code_input(
                    output["code"], 
                    language
                )
            
            # Add metadata
            processed_result = {
                "task": original_task,
                "output": output,
                "valid": schema_valid,
                "metadata": {
                    "model_name": self.model_config.get("name", "unknown"),
                    "generation_time": result.get("generation_time"),
                    "token_count": len(str(output).split()) if output else 0
                }
            }
            
            return processed_result
            
        except Exception as e:
            raise NodeError(f"Failed to process generation result: {e}")
    
    def _validate_input(self, input_data: Any) -> None:
        """
        Validate input data for generation.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValidationError: If input is invalid
        """
        super()._validate_input(input_data)
        
        if not isinstance(input_data, str):
            raise ValidationError("Input must be a string task description")
        
        if not input_data.strip():
            raise ValidationError("Task description cannot be empty")
        
        if len(input_data) > 10000:
            raise ValidationError("Task description too long (max 10000 characters)")
    
    def _validate_output(self, output_data: Dict[str, Any]) -> None:
        """
        Validate output data from generation.
        
        Args:
            output_data: Output data to validate
            
        Raises:
            ValidationError: If output is invalid
        """
        super()._validate_output(output_data)
        
        required_fields = ["task", "output", "valid"]
        for field in required_fields:
            if field not in output_data:
                raise ValidationError(f"Output missing required field: {field}")
        
        if not isinstance(output_data["output"], dict):
            raise ValidationError("Output field must be a dictionary")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation-specific statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        base_stats = self.get_stats()
        
        # Add generation-specific stats
        base_stats.update({
            "model_name": self.model_config.get("name", "unknown"),
            "model_engine": self.model_config.get("engine", "unknown"),
            "schema_validation": True,
            "processing_history_size": len(self.get_processing_history())
        })
        
        return base_stats
    
    def cleanup(self) -> None:
        """Cleanup generation node resources."""
        super().cleanup()
        # Model cleanup is handled by the factory
        logger.info("GenerateNode cleanup completed")