#!/usr/bin/env python3
"""
Architecture Summary Generator

This script automatically generates and updates architecture documentation
based on the current codebase structure and LPRA implementation.
"""

import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)


class CodeAnalyzer:
    """Analyzes the codebase to extract architectural information."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.modules = {}
        self.classes = {}
        self.functions = {}
        self.imports = {}
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase and extract architectural information."""
        logger.info("Starting codebase analysis...")
        
        python_files = list(self.root_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                self._analyze_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return self._compile_analysis_results()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during analysis."""
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
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            relative_path = file_path.relative_to(self.root_path)
            
            # Extract module information
            module_info = {
                "path": str(relative_path),
                "classes": [],
                "functions": [],
                "imports": [],
                "docstring": ast.get_docstring(tree),
                "line_count": len(content.splitlines())
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node, relative_path)
                    module_info["classes"].append(class_info)
                    self.classes[f"{relative_path}:{node.name}"] = class_info
                
                elif isinstance(node, ast.FunctionDef):
                    if not self._is_method(node, tree):  # Only top-level functions
                        func_info = self._extract_function_info(node, relative_path)
                        module_info["functions"].append(func_info)
                        self.functions[f"{relative_path}:{node.name}"] = func_info
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._extract_import_info(node)
                    module_info["imports"].extend(import_info)
            
            self.modules[str(relative_path)] = module_info
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
    
    def _extract_class_info(self, node: ast.ClassDef, file_path: Path) -> Dict[str, Any]:
        """Extract information about a class."""
        return {
            "name": node.name,
            "file": str(file_path),
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "bases": [self._get_name(base) for base in node.bases],
            "methods": [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
            "is_abstract": any(
                isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
                for method in node.body if isinstance(method, ast.FunctionDef)
                for decorator in method.decorator_list
            )
        }
    
    def _extract_function_info(self, node: ast.FunctionDef, file_path: Path) -> Dict[str, Any]:
        """Extract information about a function."""
        return {
            "name": node.name,
            "file": str(file_path),
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "args": [arg.arg for arg in node.args.args],
            "decorators": [self._get_name(decorator) for decorator in node.decorator_list],
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _extract_import_info(self, node) -> List[Dict[str, Any]]:
        """Extract import information."""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "type": "from_import",
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno
                })
        
        return imports
    
    def _get_name(self, node) -> str:
        """Get the name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)
    
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
    
    def _compile_analysis_results(self) -> Dict[str, Any]:
        """Compile the analysis results into a structured format."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_modules": len(self.modules),
            "total_classes": len(self.classes),
            "total_functions": len(self.functions),
            "modules": self.modules,
            "classes": self.classes,
            "functions": self.functions,
            "architecture_layers": self._identify_architecture_layers(),
            "dependencies": self._analyze_dependencies(),
            "complexity_metrics": self._compute_complexity_metrics()
        }
    
    def _identify_architecture_layers(self) -> Dict[str, List[str]]:
        """Identify LPRA architecture layers from the codebase."""
        layers = {
            "semantic_graph": [],
            "structured_state": [],
            "surface_context": [],
            "orchestrator": [],
            "configuration": [],
            "security": [],
            "utilities": []
        }
        
        for module_path in self.modules.keys():
            if "semantic_graph" in module_path:
                layers["semantic_graph"].append(module_path)
            elif "structured_state" in module_path:
                layers["structured_state"].append(module_path)
            elif "surface_context" in module_path:
                layers["surface_context"].append(module_path)
            elif "orchestrator" in module_path:
                layers["orchestrator"].append(module_path)
            elif "config" in module_path:
                layers["configuration"].append(module_path)
            elif "security" in module_path:
                layers["security"].append(module_path)
            elif "utils" in module_path:
                layers["utilities"].append(module_path)
        
        return layers
    
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze dependencies between modules."""
        dependencies = {}
        
        for module_path, module_info in self.modules.items():
            deps = []
            for import_info in module_info["imports"]:
                if import_info["type"] == "from_import":
                    # Check if it's an internal import
                    if not import_info["module"].startswith((".", "utils", "config", "memory", "orchestrator", "security")):
                        continue
                    deps.append(import_info["module"])
                elif import_info["type"] == "import":
                    if "." in import_info["module"]:  # Likely internal
                        deps.append(import_info["module"])
            
            dependencies[module_path] = list(set(deps))
        
        return dependencies
    
    def _compute_complexity_metrics(self) -> Dict[str, Any]:
        """Compute basic complexity metrics."""
        total_lines = sum(module["line_count"] for module in self.modules.values())
        avg_lines_per_module = total_lines / len(self.modules) if self.modules else 0
        
        class_methods = []
        for class_info in self.classes.values():
            class_methods.append(len(class_info["methods"]))
        
        avg_methods_per_class = sum(class_methods) / len(class_methods) if class_methods else 0
        
        return {
            "total_lines_of_code": total_lines,
            "average_lines_per_module": avg_lines_per_module,
            "average_methods_per_class": avg_methods_per_class,
            "largest_module": max(self.modules.items(), key=lambda x: x[1]["line_count"])[0] if self.modules else None,
            "most_complex_class": max(self.classes.items(), key=lambda x: len(x[1]["methods"]))[0] if self.classes else None
        }


class MermaidGenerator:
    """Generates Mermaid diagrams from architectural analysis."""
    
    def __init__(self, analysis_data: Dict[str, Any]):
        self.analysis_data = analysis_data
    
    def generate_architecture_diagram(self) -> str:
        """Generate a Mermaid architecture diagram."""
        layers = self.analysis_data["architecture_layers"]
        
        diagram_parts = [
            "```mermaid",
            "flowchart TD",
            "    %% Auto-generated LPRA Architecture Diagram",
            f"    %% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"    %% Total Modules: {self.analysis_data['total_modules']}",
            f"    %% Total Classes: {self.analysis_data['total_classes']}",
            "",
        ]
        
        # Generate layer subgraphs
        layer_colors = {
            "semantic_graph": "#e1f5fe",
            "structured_state": "#f3e5f5", 
            "surface_context": "#e8f5e8",
            "orchestrator": "#fff3e0",
            "configuration": "#fce4ec",
            "security": "#ffebee",
            "utilities": "#f5f5f5"
        }
        
        node_id = 1
        layer_nodes = {}
        
        for layer_name, modules in layers.items():
            if not modules:
                continue
            
            layer_display = layer_name.replace("_", " ").title()
            diagram_parts.append(f"    subgraph {layer_name}[{layer_display}]")
            
            layer_nodes[layer_name] = []
            for module in modules:
                module_name = Path(module).stem
                node_name = f"N{node_id}"
                layer_nodes[layer_name].append(node_name)
                
                # Get module info
                module_info = self.analysis_data["modules"].get(module, {})
                class_count = len(module_info.get("classes", []))
                func_count = len(module_info.get("functions", []))
                
                display_text = f"{module_name}<br/>Classes: {class_count}<br/>Functions: {func_count}"
                diagram_parts.append(f"        {node_name}[{display_text}]")
                node_id += 1
            
            diagram_parts.append("    end")
            diagram_parts.append("")
        
        # Add dependencies
        dependencies = self.analysis_data["dependencies"]
        diagram_parts.append("    %% Dependencies")
        
        for module, deps in dependencies.items():
            source_layer = self._find_module_layer(module, layers)
            if not source_layer or source_layer not in layer_nodes:
                continue
            
            for dep in deps:
                target_layer = self._find_module_layer(dep, layers)
                if target_layer and target_layer in layer_nodes:
                    # Add dependency arrow between layers
                    if source_layer != target_layer:
                        diagram_parts.append(f"    {source_layer} --> {target_layer}")
        
        # Add styling
        diagram_parts.extend([
            "",
            "    %% Styling",
            "    classDef layer1 fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef layer2 fill:#f3e5f5,stroke:#4a148c,stroke-width:2px", 
            "    classDef layer3 fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
            "    classDef core fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef security fill:#ffebee,stroke:#c62828,stroke-width:2px",
            "    classDef utils fill:#f5f5f5,stroke:#424242,stroke-width:2px",
            "",
            "```"
        ])
        
        return "\n".join(diagram_parts)
    
    def _find_module_layer(self, module_path: str, layers: Dict[str, List[str]]) -> str:
        """Find which layer a module belongs to."""
        for layer_name, modules in layers.items():
            if module_path in modules:
                return layer_name
        return ""


class DocumentationUpdater:
    """Updates architecture documentation files."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.architecture_path = root_path / "architecture"
    
    def update_documentation(self, analysis_data: Dict[str, Any], mermaid_diagram: str) -> None:
        """Update all architecture documentation."""
        logger.info("Updating architecture documentation...")
        
        # Update architecture summary
        self._update_architecture_summary(analysis_data)
        
        # Update Mermaid diagram
        self._update_mermaid_diagram(mermaid_diagram)
        
        # Update changelog
        self._update_changelog(analysis_data)
        
        logger.info("Documentation update completed")
    
    def _update_architecture_summary(self, analysis_data: Dict[str, Any]) -> None:
        """Update the architecture summary document."""
        summary_path = self.architecture_path / "current_state.md"
        
        content = [
            "# Current Architecture State",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Modules:** {analysis_data['total_modules']}",
            f"**Total Classes:** {analysis_data['total_classes']}",
            f"**Total Functions:** {analysis_data['total_functions']}",
            "",
            "## Architecture Layers",
            ""
        ]
        
        for layer_name, modules in analysis_data["architecture_layers"].items():
            if modules:
                content.append(f"### {layer_name.replace('_', ' ').title()}")
                content.append("")
                for module in modules:
                    module_info = analysis_data["modules"].get(module, {})
                    class_count = len(module_info.get("classes", []))
                    func_count = len(module_info.get("functions", []))
                    content.append(f"- **{module}**: {class_count} classes, {func_count} functions")
                content.append("")
        
        # Add complexity metrics
        metrics = analysis_data["complexity_metrics"]
        content.extend([
            "## Complexity Metrics",
            "",
            f"- **Total Lines of Code:** {metrics['total_lines_of_code']:,}",
            f"- **Average Lines per Module:** {metrics['average_lines_per_module']:.1f}",
            f"- **Average Methods per Class:** {metrics['average_methods_per_class']:.1f}",
            f"- **Largest Module:** {metrics['largest_module']}",
            f"- **Most Complex Class:** {metrics['most_complex_class']}",
            ""
        ])
        
        with open(summary_path, 'w') as f:
            f.write("\n".join(content))
        
        logger.info(f"Updated architecture summary: {summary_path}")
    
    def _update_mermaid_diagram(self, mermaid_diagram: str) -> None:
        """Update the Mermaid diagram file."""
        diagram_path = self.architecture_path / "current_architecture.mmd"
        
        with open(diagram_path, 'w') as f:
            f.write(mermaid_diagram)
        
        logger.info(f"Updated Mermaid diagram: {diagram_path}")
    
    def _update_changelog(self, analysis_data: Dict[str, Any]) -> None:
        """Update the changelog with current state."""
        changelog_path = self.architecture_path / "changelog.md"
        
        if not changelog_path.exists():
            return
        
        # Read existing changelog
        with open(changelog_path, 'r') as f:
            existing_content = f.read()
        
        # Create new entry
        new_entry = [
            f"## Auto-Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "### Current State",
            f"- Total modules: {analysis_data['total_modules']}",
            f"- Total classes: {analysis_data['total_classes']}",
            f"- Total functions: {analysis_data['total_functions']}",
            f"- Lines of code: {analysis_data['complexity_metrics']['total_lines_of_code']:,}",
            "",
            "### Architecture Layers",
            ""
        ]
        
        for layer_name, modules in analysis_data["architecture_layers"].items():
            if modules:
                new_entry.append(f"- **{layer_name}**: {len(modules)} modules")
        
        new_entry.extend(["", "---", ""])
        
        # Insert new entry after the first header
        lines = existing_content.split('\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('## ') and i > 0:
                insert_index = i
                break
        
        if insert_index > 0:
            updated_lines = lines[:insert_index] + new_entry + lines[insert_index:]
            with open(changelog_path, 'w') as f:
                f.write('\n'.join(updated_lines))
            
            logger.info(f"Updated changelog: {changelog_path}")


def main():
    """Main function to generate architecture summary."""
    parser = argparse.ArgumentParser(description="Generate LPRA architecture summary")
    parser.add_argument("--root", type=Path, default=Path.cwd(), 
                       help="Root directory of the project")
    parser.add_argument("--output", type=Path, 
                       help="Output directory for generated files")
    parser.add_argument("--update-docs", action="store_true",
                       help="Update documentation files")
    
    args = parser.parse_args()
    
    try:
        # Analyze codebase
        analyzer = CodeAnalyzer(args.root)
        analysis_data = analyzer.analyze_codebase()
        
        # Generate Mermaid diagram
        mermaid_generator = MermaidGenerator(analysis_data)
        mermaid_diagram = mermaid_generator.generate_architecture_diagram()
        
        # Output results
        if args.output:
            output_path = args.output
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save analysis data
            with open(output_path / "analysis.json", 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            # Save Mermaid diagram
            with open(output_path / "architecture.mmd", 'w') as f:
                f.write(mermaid_diagram)
            
            logger.info(f"Generated files saved to {output_path}")
        
        # Update documentation if requested
        if args.update_docs:
            updater = DocumentationUpdater(args.root)
            updater.update_documentation(analysis_data, mermaid_diagram)
        
        # Print summary
        print(f"\n🏗️  Architecture Analysis Complete")
        print(f"📊 Modules: {analysis_data['total_modules']}")
        print(f"🏛️  Classes: {analysis_data['total_classes']}")
        print(f"⚙️  Functions: {analysis_data['total_functions']}")
        print(f"📝 Lines of Code: {analysis_data['complexity_metrics']['total_lines_of_code']:,}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Architecture generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())