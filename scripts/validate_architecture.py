#!/usr/bin/env python3
"""
Architecture Validation Script

This script validates the LPRA architecture implementation against the
defined blueprint and cognitive contract. It checks for consistency,
completeness, and compliance with architectural principles.
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import argparse
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)


class ValidationResult:
    """Represents the result of a validation check."""
    
    def __init__(self, check_name: str, passed: bool, message: str, severity: str = "ERROR"):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.severity = severity  # ERROR, WARNING, INFO
        self.timestamp = datetime.now()
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else f"❌ {self.severity}"
        return f"{status}: {self.check_name} - {self.message}"


class ArchitectureValidator:
    """Validates LPRA architecture implementation."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.results: List[ValidationResult] = []
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load LPRA configuration."""
        config_path = self.root_path / "config" / "lpra.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks."""
        logger.info("Starting comprehensive architecture validation...")
        
        # Core structure validation
        self._validate_directory_structure()
        self._validate_required_files()
        self._validate_mermaid_diagram()
        
        # Layer implementation validation
        self._validate_semantic_graph_layer()
        self._validate_structured_state_layer()
        self._validate_surface_context_layer()
        
        # Integration validation
        self._validate_configuration_integration()
        self._validate_orchestrator_integration()
        
        # Documentation validation
        self._validate_documentation_consistency()
        self._validate_cognitive_contract()
        
        # Code quality validation
        self._validate_code_quality()
        self._validate_test_coverage()
        
        logger.info(f"Validation completed with {len(self.results)} checks")
        return self.results
    
    def _validate_directory_structure(self) -> None:
        """Validate the expected directory structure."""
        required_dirs = [
            "architecture",
            "config", 
            "memory",
            "orchestrator",
            "security",
            "utils",
            "tests",
            "scripts"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.root_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.results.append(ValidationResult(
                    f"Directory Structure: {dir_name}",
                    True,
                    f"Required directory {dir_name} exists"
                ))
            else:
                self.results.append(ValidationResult(
                    f"Directory Structure: {dir_name}",
                    False,
                    f"Required directory {dir_name} is missing"
                ))
    
    def _validate_required_files(self) -> None:
        """Validate that required files exist."""
        required_files = [
            "architecture/LPRA.mmd",
            "architecture/LPRA.md", 
            "architecture/changelog.md",
            "config/lpra.yaml",
            "memory/semantic_graph.py",
            "memory/structured_state.py",
            "memory/surface_context.py",
            "main.py",
            "requirements.txt"
        ]
        
        for file_path in required_files:
            full_path = self.root_path / file_path
            if full_path.exists() and full_path.is_file():
                self.results.append(ValidationResult(
                    f"Required File: {file_path}",
                    True,
                    f"Required file {file_path} exists"
                ))
            else:
                self.results.append(ValidationResult(
                    f"Required File: {file_path}",
                    False,
                    f"Required file {file_path} is missing"
                ))
    
    def _validate_mermaid_diagram(self) -> None:
        """Validate the Mermaid architecture diagram."""
        mermaid_path = self.root_path / "architecture" / "LPRA.mmd"
        
        if not mermaid_path.exists():
            self.results.append(ValidationResult(
                "Mermaid Diagram",
                False,
                "LPRA.mmd file is missing"
            ))
            return
        
        try:
            with open(mermaid_path, 'r') as f:
                content = f.read()
            
            # Check for required elements
            required_elements = [
                "Semantic Graph Layer",
                "Structured State Layer", 
                "Surface Context Layer",
                "flowchart TD",
                "subgraph"
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                self.results.append(ValidationResult(
                    "Mermaid Diagram Content",
                    False,
                    f"Missing required elements: {', '.join(missing_elements)}",
                    "WARNING"
                ))
            else:
                self.results.append(ValidationResult(
                    "Mermaid Diagram Content",
                    True,
                    "All required diagram elements present"
                ))
            
            # Check diagram syntax
            if content.strip().startswith("```mermaid") and content.strip().endswith("```"):
                self.results.append(ValidationResult(
                    "Mermaid Diagram Syntax",
                    True,
                    "Proper Mermaid code block syntax"
                ))
            else:
                self.results.append(ValidationResult(
                    "Mermaid Diagram Syntax",
                    False,
                    "Invalid Mermaid code block syntax",
                    "WARNING"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                "Mermaid Diagram",
                False,
                f"Error reading Mermaid diagram: {e}"
            ))
    
    def _validate_semantic_graph_layer(self) -> None:
        """Validate Layer 1: Semantic Graph implementation."""
        graph_file = self.root_path / "memory" / "semantic_graph.py"
        
        if not graph_file.exists():
            self.results.append(ValidationResult(
                "Semantic Graph Layer",
                False,
                "semantic_graph.py is missing"
            ))
            return
        
        try:
            with open(graph_file, 'r') as f:
                content = f.read()
            
            # Check for required classes
            required_classes = [
                "SemanticGraph",
                "SemanticNode", 
                "SemanticEdge",
                "NodeType",
                "EdgeType",
                "GraphMechanism"
            ]
            
            missing_classes = []
            for class_name in required_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)
            
            if missing_classes:
                self.results.append(ValidationResult(
                    "Semantic Graph Classes",
                    False,
                    f"Missing required classes: {', '.join(missing_classes)}"
                ))
            else:
                self.results.append(ValidationResult(
                    "Semantic Graph Classes",
                    True,
                    "All required classes implemented"
                ))
            
            # Check for required mechanisms
            required_mechanisms = [
                "TimeWeightedDecay",
                "ReinforcementLearning",
                "PrincipledPruning",
                "SemanticClustering"
            ]
            
            missing_mechanisms = []
            for mechanism in required_mechanisms:
                if f"class {mechanism}" not in content:
                    missing_mechanisms.append(mechanism)
            
            if missing_mechanisms:
                self.results.append(ValidationResult(
                    "Graph Mechanisms",
                    False,
                    f"Missing required mechanisms: {', '.join(missing_mechanisms)}"
                ))
            else:
                self.results.append(ValidationResult(
                    "Graph Mechanisms",
                    True,
                    "All required mechanisms implemented"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                "Semantic Graph Layer",
                False,
                f"Error validating semantic graph: {e}"
            ))
    
    def _validate_structured_state_layer(self) -> None:
        """Validate Layer 2: Structured State implementation."""
        state_file = self.root_path / "memory" / "structured_state.py"
        
        if not state_file.exists():
            self.results.append(ValidationResult(
                "Structured State Layer",
                False,
                "structured_state.py is missing"
            ))
            return
        
        try:
            with open(state_file, 'r') as f:
                content = f.read()
            
            # Check for required classes
            required_classes = [
                "StructuredStateLayer",
                "CanonicalStateSchema",
                "DerivationEngine",
                "StateStore",
                "SQLiteStateStore",
                "LanceDBStateStore"
            ]
            
            missing_classes = []
            for class_name in required_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)
            
            if missing_classes:
                self.results.append(ValidationResult(
                    "Structured State Classes",
                    False,
                    f"Missing required classes: {', '.join(missing_classes)}"
                ))
            else:
                self.results.append(ValidationResult(
                    "Structured State Classes",
                    True,
                    "All required classes implemented"
                ))
            
            # Check for Pydantic usage
            if "from pydantic import" in content:
                self.results.append(ValidationResult(
                    "State Schema Validation",
                    True,
                    "Pydantic validation implemented"
                ))
            else:
                self.results.append(ValidationResult(
                    "State Schema Validation",
                    False,
                    "Pydantic validation not found",
                    "WARNING"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                "Structured State Layer",
                False,
                f"Error validating structured state: {e}"
            ))
    
    def _validate_surface_context_layer(self) -> None:
        """Validate Layer 3: Surface Context implementation."""
        context_file = self.root_path / "memory" / "surface_context.py"
        
        if not context_file.exists():
            self.results.append(ValidationResult(
                "Surface Context Layer",
                False,
                "surface_context.py is missing"
            ))
            return
        
        try:
            with open(context_file, 'r') as f:
                content = f.read()
            
            # Check for required classes
            required_classes = [
                "SurfaceContextLayer",
                "ContextCompressor",
                "RelevanceBasedCompressor",
                "TemporalSlidingCompressor",
                "HierarchicalCompressor",
                "HumanSummaryGenerator"
            ]
            
            missing_classes = []
            for class_name in required_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)
            
            if missing_classes:
                self.results.append(ValidationResult(
                    "Surface Context Classes",
                    False,
                    f"Missing required classes: {', '.join(missing_classes)}"
                ))
            else:
                self.results.append(ValidationResult(
                    "Surface Context Classes",
                    True,
                    "All required classes implemented"
                ))
            
            # Check for compression strategies
            compression_strategies = [
                "RELEVANCE_BASED",
                "TEMPORAL_SLIDING", 
                "HIERARCHICAL"
            ]
            
            missing_strategies = []
            for strategy in compression_strategies:
                if strategy not in content:
                    missing_strategies.append(strategy)
            
            if missing_strategies:
                self.results.append(ValidationResult(
                    "Compression Strategies",
                    False,
                    f"Missing compression strategies: {', '.join(missing_strategies)}",
                    "WARNING"
                ))
            else:
                self.results.append(ValidationResult(
                    "Compression Strategies",
                    True,
                    "All compression strategies implemented"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                "Surface Context Layer",
                False,
                f"Error validating surface context: {e}"
            ))
    
    def _validate_configuration_integration(self) -> None:
        """Validate configuration integration."""
        config_file = self.root_path / "config" / "lpra.yaml"
        
        if not config_file.exists():
            self.results.append(ValidationResult(
                "LPRA Configuration",
                False,
                "lpra.yaml configuration file is missing"
            ))
            return
        
        try:
            # Check configuration sections
            required_sections = [
                "semantic_graph",
                "structured_state",
                "surface_context",
                "integration",
                "performance",
                "security"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in self.config:
                    missing_sections.append(section)
            
            if missing_sections:
                self.results.append(ValidationResult(
                    "Configuration Sections",
                    False,
                    f"Missing configuration sections: {', '.join(missing_sections)}"
                ))
            else:
                self.results.append(ValidationResult(
                    "Configuration Sections",
                    True,
                    "All required configuration sections present"
                ))
            
            # Validate cognitive contract parameters
            if "semantic_graph" in self.config:
                sg_config = self.config["semantic_graph"]
                required_params = [
                    "max_graph_nodes",
                    "success_boost",
                    "base_decay_rate",
                    "edge_strength_threshold"
                ]
                
                missing_params = []
                for param in required_params:
                    if param not in sg_config:
                        missing_params.append(param)
                
                if missing_params:
                    self.results.append(ValidationResult(
                        "Cognitive Contract Parameters",
                        False,
                        f"Missing semantic graph parameters: {', '.join(missing_params)}",
                        "WARNING"
                    ))
                else:
                    self.results.append(ValidationResult(
                        "Cognitive Contract Parameters",
                        True,
                        "Semantic graph parameters complete"
                    ))
                    
        except Exception as e:
            self.results.append(ValidationResult(
                "Configuration Integration",
                False,
                f"Error validating configuration: {e}"
            ))
    
    def _validate_orchestrator_integration(self) -> None:
        """Validate orchestrator integration with LPRA."""
        main_file = self.root_path / "main.py"
        
        if not main_file.exists():
            self.results.append(ValidationResult(
                "Main Integration",
                False,
                "main.py is missing"
            ))
            return
        
        try:
            with open(main_file, 'r') as f:
                content = f.read()
            
            # Check for LPRA imports
            lpra_imports = [
                "semantic_graph",
                "structured_state", 
                "surface_context"
            ]
            
            missing_imports = []
            for import_name in lpra_imports:
                if import_name not in content:
                    missing_imports.append(import_name)
            
            if missing_imports:
                self.results.append(ValidationResult(
                    "LPRA Integration Imports",
                    False,
                    f"Missing LPRA imports in main.py: {', '.join(missing_imports)}",
                    "WARNING"
                ))
            else:
                self.results.append(ValidationResult(
                    "LPRA Integration Imports",
                    True,
                    "LPRA imports present in main.py"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                "Orchestrator Integration",
                False,
                f"Error validating orchestrator integration: {e}"
            ))
    
    def _validate_documentation_consistency(self) -> None:
        """Validate documentation consistency."""
        lpra_doc = self.root_path / "architecture" / "LPRA.md"
        
        if not lpra_doc.exists():
            self.results.append(ValidationResult(
                "LPRA Documentation",
                False,
                "LPRA.md documentation is missing"
            ))
            return
        
        try:
            with open(lpra_doc, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = [
                "Architecture Layers",
                "Layer 1: Semantic Graph Layer",
                "Layer 2: Structured State Layer",
                "Layer 3: Surface Context Layer",
                "Cognitive Contract",
                "Implementation Roadmap"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                self.results.append(ValidationResult(
                    "Documentation Sections",
                    False,
                    f"Missing documentation sections: {', '.join(missing_sections)}",
                    "WARNING"
                ))
            else:
                self.results.append(ValidationResult(
                    "Documentation Sections",
                    True,
                    "All required documentation sections present"
                ))
            
            # Check documentation freshness
            if "2025-11-23" in content:
                self.results.append(ValidationResult(
                    "Documentation Freshness",
                    True,
                    "Documentation appears to be current"
                ))
            else:
                self.results.append(ValidationResult(
                    "Documentation Freshness",
                    False,
                    "Documentation may be outdated",
                    "WARNING"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                "Documentation Consistency",
                False,
                f"Error validating documentation: {e}"
            ))
    
    def _validate_cognitive_contract(self) -> None:
        """Validate cognitive contract compliance."""
        if not self.config:
            self.results.append(ValidationResult(
                "Cognitive Contract",
                False,
                "No configuration loaded for cognitive contract validation"
            ))
            return
        
        # Check parameter ranges
        if "semantic_graph" in self.config:
            sg_config = self.config["semantic_graph"]
            
            # Validate reinforcement parameters
            success_boost = sg_config.get("success_boost", 0)
            if 1.0 <= success_boost <= 2.0:
                self.results.append(ValidationResult(
                    "Success Boost Parameter",
                    True,
                    f"Success boost {success_boost} is within valid range"
                ))
            else:
                self.results.append(ValidationResult(
                    "Success Boost Parameter",
                    False,
                    f"Success boost {success_boost} is outside valid range [1.0, 2.0]",
                    "WARNING"
                ))
            
            # Validate decay parameters
            base_decay = sg_config.get("base_decay_rate", 0)
            if 0.8 <= base_decay <= 1.0:
                self.results.append(ValidationResult(
                    "Base Decay Rate",
                    True,
                    f"Base decay rate {base_decay} is within valid range"
                ))
            else:
                self.results.append(ValidationResult(
                    "Base Decay Rate",
                    False,
                    f"Base decay rate {base_decay} is outside valid range [0.8, 1.0]",
                    "WARNING"
                ))
    
    def _validate_code_quality(self) -> None:
        """Validate code quality standards."""
        python_files = list(self.root_path.rglob("*.py"))
        
        if not python_files:
            self.results.append(ValidationResult(
                "Code Quality",
                False,
                "No Python files found for quality validation"
            ))
            return
        
        # Check for basic quality indicators
        total_files = len(python_files)
        files_with_docstrings = 0
        files_with_type_hints = 0
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for docstrings
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1
                
                # Check for type hints
                if "from typing import" in content or ": " in content:
                    files_with_type_hints += 1
                    
            except Exception:
                continue
        
        # Calculate percentages
        docstring_percentage = (files_with_docstrings / total_files) * 100
        type_hint_percentage = (files_with_type_hints / total_files) * 100
        
        # Validate docstring coverage
        if docstring_percentage >= 80:
            self.results.append(ValidationResult(
                "Docstring Coverage",
                True,
                f"Good docstring coverage: {docstring_percentage:.1f}%"
            ))
        elif docstring_percentage >= 50:
            self.results.append(ValidationResult(
                "Docstring Coverage",
                False,
                f"Moderate docstring coverage: {docstring_percentage:.1f}%",
                "WARNING"
            ))
        else:
            self.results.append(ValidationResult(
                "Docstring Coverage",
                False,
                f"Low docstring coverage: {docstring_percentage:.1f}%"
            ))
        
        # Validate type hint coverage
        if type_hint_percentage >= 70:
            self.results.append(ValidationResult(
                "Type Hint Coverage",
                True,
                f"Good type hint coverage: {type_hint_percentage:.1f}%"
            ))
        else:
            self.results.append(ValidationResult(
                "Type Hint Coverage",
                False,
                f"Low type hint coverage: {type_hint_percentage:.1f}%",
                "WARNING"
            ))
    
    def _validate_test_coverage(self) -> None:
        """Validate test coverage."""
        test_dir = self.root_path / "tests"
        
        if not test_dir.exists():
            self.results.append(ValidationResult(
                "Test Directory",
                False,
                "Tests directory is missing"
            ))
            return
        
        test_files = list(test_dir.glob("test_*.py"))
        
        if not test_files:
            self.results.append(ValidationResult(
                "Test Files",
                False,
                "No test files found in tests directory"
            ))
        else:
            self.results.append(ValidationResult(
                "Test Files",
                True,
                f"Found {len(test_files)} test files"
            ))
        
        # Check for test configuration
        pytest_config = self.root_path / "pytest.ini"
        if pytest_config.exists():
            self.results.append(ValidationResult(
                "Test Configuration",
                True,
                "pytest.ini configuration found"
            ))
        else:
            self.results.append(ValidationResult(
                "Test Configuration",
                False,
                "pytest.ini configuration is missing",
                "WARNING"
            ))
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during validation."""
        skip_patterns = [
            "__pycache__",
            ".git",
            "venv",
            "env",
            ".pytest_cache",
            "build",
            "dist"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def generate_report(self) -> str:
        """Generate a validation report."""
        passed_checks = sum(1 for r in self.results if r.passed)
        total_checks = len(self.results)
        
        report_lines = [
            "🔍 LPRA Architecture Validation Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Checks: {total_checks}",
            f"Passed: {passed_checks}",
            f"Failed: {total_checks - passed_checks}",
            f"Success Rate: {(passed_checks / total_checks * 100):.1f}%",
            "",
            "📋 Detailed Results:",
            ""
        ]
        
        # Group results by category
        categories = {}
        for result in self.results:
            category = result.check_name.split(":")[0] if ":" in result.check_name else "General"
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category, results in categories.items():
            report_lines.append(f"## {category}")
            report_lines.append("")
            
            for result in results:
                report_lines.append(str(result))
            
            report_lines.append("")
        
        # Summary and recommendations
        failed_results = [r for r in self.results if not r.passed]
        if failed_results:
            report_lines.extend([
                "🚨 Issues Found:",
                ""
            ])
            
            for result in failed_results:
                if result.severity == "ERROR":
                    report_lines.append(f"❌ CRITICAL: {result.check_name} - {result.message}")
            
            for result in failed_results:
                if result.severity == "WARNING":
                    report_lines.append(f"⚠️  WARNING: {result.check_name} - {result.message}")
        else:
            report_lines.extend([
                "🎉 All Validation Checks Passed!",
                "The LPRA architecture implementation is compliant with the blueprint.",
                ""
            ])
        
        return "\n".join(report_lines)


def main():
    """Main function to run architecture validation."""
    parser = argparse.ArgumentParser(description="Validate LPRA architecture implementation")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                       help="Root directory of the project")
    parser.add_argument("--output", type=Path,
                       help="Output file for validation report")
    parser.add_argument("--fail-on-error", action="store_true",
                       help="Exit with error code if validation fails")
    
    args = parser.parse_args()
    
    try:
        # Run validation
        validator = ArchitectureValidator(args.root)
        results = validator.validate_all()
        
        # Generate report
        report = validator.generate_report()
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {args.output}")
        else:
            print(report)
        
        # Check for failures
        failed_checks = sum(1 for r in results if not r.passed)
        critical_failures = sum(1 for r in results if not r.passed and r.severity == "ERROR")
        
        if args.fail_on_error and critical_failures > 0:
            logger.error(f"Validation failed with {critical_failures} critical errors")
            return 1
        
        if failed_checks > 0:
            logger.warning(f"Validation completed with {failed_checks} issues")
        else:
            logger.info("All validation checks passed!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())