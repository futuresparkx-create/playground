"""
LPRA Integration Layer

This module provides the integration layer that connects the LPRA architecture
with the existing playground system. It manages the lifecycle of all three layers
and provides a unified interface for the orchestrator.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from memory.semantic_graph import (
    SemanticGraph, SemanticNode, SemanticEdge, NodeType, EdgeType,
    NodeMetadata, EdgeMetadata
)
from memory.structured_state import (
    StructuredStateLayer, CanonicalStateSchema, ConsistencyLevel
)
from memory.surface_context import (
    SurfaceContextLayer, ContextType, CompressionStrategy, AttentionWeights
)
from utils.logging_config import get_logger
from utils.exceptions import PlaygroundException

logger = get_logger(__name__)


class LPRAManager:
    """
    Central manager for the LPRA (Long-term Persistent Reasoning Architecture).
    
    Coordinates all three layers and provides a unified interface for the
    orchestrator and other system components.
    """
    
    def __init__(self, config_path: Optional[Path] = None, data_dir: Optional[Path] = None):
        """Initialize the LPRA manager."""
        self.config_path = config_path or Path("config/lpra.yaml")
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize layers
        self.semantic_graph = SemanticGraph(self.config.get("semantic_graph", {}))
        
        self.structured_state = StructuredStateLayer(
            sqlite_path=self.data_dir / "lpra_states.db",
            lancedb_path=self.data_dir / "lpra_vectors",
            consistency_level=ConsistencyLevel(
                self.config.get("structured_state", {}).get("consistency_level", "strict")
            )
        )
        
        self.surface_context = SurfaceContextLayer(
            config=self.config.get("surface_context", {})
        )
        
        # Background maintenance
        self._maintenance_thread = None
        self._shutdown_event = threading.Event()
        self._last_maintenance = datetime.now()
        
        # Statistics
        self.stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "states_computed": 0,
            "contexts_generated": 0,
            "maintenance_cycles": 0,
            "started_at": datetime.now()
        }
        
        logger.info("LPRA Manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load LPRA configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"LPRA config not found at {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LPRA configuration."""
        return {
            "semantic_graph": {
                "max_graph_nodes": 10000,
                "success_boost": 1.2,
                "base_decay_rate": 0.95,
                "edge_strength_threshold": 0.2,
                "node_relevance_threshold": 0.15
            },
            "structured_state": {
                "consistency_level": "strict",
                "max_stored_states": 1000
            },
            "surface_context": {
                "max_context_tokens": 4000,
                "relevance_threshold": 0.5,
                "cache_ttl_minutes": 30
            },
            "performance": {
                "maintenance_interval_minutes": 60
            }
        }
    
    def start_background_maintenance(self) -> None:
        """Start background maintenance thread."""
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            logger.warning("Background maintenance already running")
            return
        
        self._shutdown_event.clear()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="LPRA-Maintenance"
        )
        self._maintenance_thread.start()
        logger.info("Started LPRA background maintenance")
    
    def stop_background_maintenance(self) -> None:
        """Stop background maintenance thread."""
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._shutdown_event.set()
            self._maintenance_thread.join(timeout=10)
            logger.info("Stopped LPRA background maintenance")
    
    def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        interval_minutes = self.config.get("performance", {}).get("maintenance_interval_minutes", 60)
        interval_seconds = interval_minutes * 60
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for interval or shutdown signal
                if self._shutdown_event.wait(timeout=interval_seconds):
                    break  # Shutdown requested
                
                # Run maintenance
                self._run_maintenance()
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                # Continue running despite errors
    
    def _run_maintenance(self) -> None:
        """Run maintenance tasks on all layers."""
        logger.info("Running LPRA maintenance cycle")
        start_time = time.time()
        
        try:
            # Layer 1: Apply graph mechanisms
            self.semantic_graph.apply_mechanisms()
            
            # Layer 2: Cleanup old states
            cleaned_states = self.structured_state.cleanup_old_states(
                keep_count=self.config.get("structured_state", {}).get("max_stored_states", 1000)
            )
            
            # Layer 3: Clear expired cache
            cleared_contexts = self.surface_context.clear_cache()
            
            # Update statistics
            self.stats["maintenance_cycles"] += 1
            self._last_maintenance = datetime.now()
            
            duration = time.time() - start_time
            logger.info(f"Maintenance completed in {duration:.2f}s: "
                       f"cleaned {cleaned_states} states, {cleared_contexts} contexts")
            
        except Exception as e:
            logger.error(f"Maintenance cycle failed: {e}")
    
    # === ORCHESTRATOR INTEGRATION METHODS ===
    
    def record_task_start(self, task_description: str, task_id: Optional[str] = None) -> str:
        """Record the start of a new task in the semantic graph."""
        try:
            # Create task node
            task_node = SemanticNode(
                type=NodeType.OBJECTIVE,
                name=f"Task: {task_description[:50]}...",
                description=task_description,
                content={
                    "task_id": task_id,
                    "full_description": task_description,
                    "status": "started"
                }
            )
            
            node_id = self.semantic_graph.add_node(task_node)
            self.stats["nodes_created"] += 1
            
            logger.info(f"Recorded task start: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to record task start: {e}")
            raise PlaygroundException(f"Failed to record task: {e}")
    
    def record_generation_result(self, task_node_id: str, generation_result: Dict[str, Any]) -> str:
        """Record code generation result."""
        try:
            # Create generation node
            gen_node = SemanticNode(
                type=NodeType.OUTCOME,
                name="Code Generation",
                description="Generated code from AI model",
                content={
                    "code": generation_result.get("output", ""),
                    "model_used": generation_result.get("model", "unknown"),
                    "tokens_used": generation_result.get("tokens", 0),
                    "generation_time": generation_result.get("time", 0)
                }
            )
            
            gen_node_id = self.semantic_graph.add_node(gen_node)
            
            # Create relationship to task
            task_edge = SemanticEdge(
                source_id=task_node_id,
                target_id=gen_node_id,
                type=EdgeType.CAUSAL,
                weight=1.0,
                properties={"relationship": "task_to_generation"}
            )
            
            self.semantic_graph.add_edge(task_edge)
            self.stats["nodes_created"] += 1
            self.stats["edges_created"] += 1
            
            logger.debug(f"Recorded generation result: {gen_node_id}")
            return gen_node_id
            
        except Exception as e:
            logger.error(f"Failed to record generation result: {e}")
            return ""
    
    def record_reflection_result(self, gen_node_id: str, reflection_result: Optional[Dict[str, Any]]) -> Optional[str]:
        """Record reflection/analysis result."""
        if not reflection_result:
            return None
        
        try:
            # Create reflection node
            reflect_node = SemanticNode(
                type=NodeType.DECISION,
                name="Code Reflection",
                description="AI reflection on generated code",
                content={
                    "reflection": reflection_result.get("output", ""),
                    "suggestions": reflection_result.get("suggestions", []),
                    "quality_score": reflection_result.get("quality_score", 0)
                }
            )
            
            reflect_node_id = self.semantic_graph.add_node(reflect_node)
            
            # Create relationship to generation
            reflect_edge = SemanticEdge(
                source_id=gen_node_id,
                target_id=reflect_node_id,
                type=EdgeType.IMPROVEMENT,
                weight=0.8,
                properties={"relationship": "generation_to_reflection"}
            )
            
            self.semantic_graph.add_edge(reflect_edge)
            self.stats["nodes_created"] += 1
            self.stats["edges_created"] += 1
            
            logger.debug(f"Recorded reflection result: {reflect_node_id}")
            return reflect_node_id
            
        except Exception as e:
            logger.error(f"Failed to record reflection result: {e}")
            return None
    
    def record_test_result(self, gen_node_id: str, test_result: Dict[str, Any]) -> str:
        """Record test/validation result."""
        try:
            # Create test node
            test_node = SemanticNode(
                type=NodeType.OUTCOME,
                name="Code Testing",
                description="Static analysis and testing results",
                content={
                    "test_results": test_result.get("output", ""),
                    "passed": test_result.get("passed", False),
                    "issues_found": test_result.get("issues", []),
                    "quality_metrics": test_result.get("metrics", {})
                }
            )
            
            test_node_id = self.semantic_graph.add_node(test_node)
            
            # Create relationship to generation
            test_edge = SemanticEdge(
                source_id=gen_node_id,
                target_id=test_node_id,
                type=EdgeType.VALIDATION,
                weight=1.0,
                properties={"relationship": "generation_to_test"}
            )
            
            self.semantic_graph.add_edge(test_edge)
            self.stats["nodes_created"] += 1
            self.stats["edges_created"] += 1
            
            logger.debug(f"Recorded test result: {test_node_id}")
            return test_node_id
            
        except Exception as e:
            logger.error(f"Failed to record test result: {e}")
            return ""
    
    def record_learning_outcome(self, cycle_data: Dict[str, Any]) -> str:
        """Record learning outcome from a complete cycle."""
        try:
            # Create learning node
            learn_node = SemanticNode(
                type=NodeType.PATTERN,
                name="Learning Outcome",
                description="Insights and patterns learned from cycle",
                content={
                    "insights": cycle_data.get("learn", {}).get("output", ""),
                    "patterns_identified": cycle_data.get("learn", {}).get("patterns", []),
                    "success_indicators": cycle_data.get("learn", {}).get("success", []),
                    "cycle_summary": self._summarize_cycle(cycle_data)
                }
            )
            
            learn_node_id = self.semantic_graph.add_node(learn_node)
            
            # Connect to all cycle components
            for component_name, component_data in cycle_data.items():
                if component_name == "learn" or not isinstance(component_data, dict):
                    continue
                
                component_id = component_data.get("node_id")
                if component_id:
                    learn_edge = SemanticEdge(
                        source_id=component_id,
                        target_id=learn_node_id,
                        type=EdgeType.CAUSAL,
                        weight=0.6,
                        properties={"relationship": f"{component_name}_to_learning"}
                    )
                    self.semantic_graph.add_edge(learn_edge)
                    self.stats["edges_created"] += 1
            
            self.stats["nodes_created"] += 1
            logger.debug(f"Recorded learning outcome: {learn_node_id}")
            return learn_node_id
            
        except Exception as e:
            logger.error(f"Failed to record learning outcome: {e}")
            return ""
    
    def _summarize_cycle(self, cycle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of a complete cycle."""
        return {
            "has_generation": "generation" in cycle_data,
            "has_reflection": "reflection" in cycle_data and cycle_data["reflection"] is not None,
            "has_test": "test" in cycle_data,
            "has_learning": "learn" in cycle_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def compute_current_state(self) -> CanonicalStateSchema:
        """Compute and store the current canonical state."""
        try:
            state_id = self.structured_state.compute_and_store_state(self.semantic_graph)
            state = self.structured_state.load_state(state_id)
            
            if state:
                self.stats["states_computed"] += 1
                logger.debug(f"Computed current state: {state_id}")
                return state
            else:
                raise PlaygroundException(f"Failed to load computed state {state_id}")
                
        except Exception as e:
            logger.error(f"Failed to compute current state: {e}")
            raise PlaygroundException(f"State computation failed: {e}")
    
    def get_llm_context(self, 
                       strategy: CompressionStrategy = CompressionStrategy.RELEVANCE_BASED,
                       max_tokens: int = 4000,
                       focus_nodes: Optional[List[str]] = None) -> str:
        """Get LLM-optimized context for the current state."""
        try:
            # Get current state
            current_state = self.compute_current_state()
            
            # Create attention weights
            attention = self.surface_context.create_attention_weights(
                focus_nodes=focus_nodes,
                focus_types=[NodeType.GOAL, NodeType.OBJECTIVE, NodeType.OUTCOME]
            )
            
            # Generate context
            context_window = self.surface_context.get_llm_context(
                graph=self.semantic_graph,
                state=current_state,
                strategy=strategy,
                max_tokens=max_tokens,
                attention=attention
            )
            
            self.stats["contexts_generated"] += 1
            logger.debug(f"Generated LLM context: {context_window.token_count} tokens")
            return context_window.content
            
        except Exception as e:
            logger.error(f"Failed to get LLM context: {e}")
            return f"Error generating context: {e}"
    
    def get_human_summary(self, summary_type: str = "progress") -> str:
        """Get human-readable summary of the current state."""
        try:
            # Get current state
            current_state = self.compute_current_state()
            
            # Generate summary
            summary = self.surface_context.get_human_summary(
                graph=self.semantic_graph,
                state=current_state,
                summary_type=summary_type
            )
            
            logger.debug(f"Generated human summary: {summary_type}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get human summary: {e}")
            return f"Error generating summary: {e}"
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Graph statistics
            graph_stats = self.semantic_graph.get_statistics()
            
            # State statistics
            state_stats = self.structured_state.get_state_statistics()
            
            # Context statistics
            context_stats = self.surface_context.get_context_statistics()
            
            # Combine with manager statistics
            uptime = datetime.now() - self.stats["started_at"]
            
            return {
                "lpra_manager": {
                    **self.stats,
                    "uptime_seconds": uptime.total_seconds(),
                    "last_maintenance": self._last_maintenance.isoformat(),
                    "maintenance_thread_alive": self._maintenance_thread.is_alive() if self._maintenance_thread else False
                },
                "semantic_graph": graph_stats,
                "structured_state": state_stats,
                "surface_context": context_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get system statistics: {e}")
            return {"error": str(e)}
    
    def cleanup(self) -> None:
        """Cleanup LPRA resources."""
        logger.info("Starting LPRA cleanup...")
        
        try:
            # Stop background maintenance
            self.stop_background_maintenance()
            
            # Final maintenance run
            self._run_maintenance()
            
            # Clear caches
            self.surface_context.clear_cache()
            
            logger.info("LPRA cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during LPRA cleanup: {e}")


class LPRAIntegratedGraph:
    """
    Enhanced ImprovementGraph that integrates with LPRA.
    
    This class wraps the existing ImprovementGraph and adds LPRA functionality
    while maintaining backward compatibility.
    """
    
    def __init__(self, config: Dict[str, Any], lpra_manager: Optional[LPRAManager] = None):
        """Initialize the LPRA-integrated graph."""
        # Import here to avoid circular imports
        from orchestrator.graph import ImprovementGraph
        
        self.original_graph = ImprovementGraph(config)
        self.lpra_manager = lpra_manager or LPRAManager()
        self.config = config
        
        # Start background maintenance
        self.lpra_manager.start_background_maintenance()
        
        logger.info("LPRA-integrated graph initialized")
    
    def cycle(self, input_task: str) -> Dict[str, Any]:
        """Run an improvement cycle with LPRA integration."""
        try:
            # Record task start in LPRA
            task_node_id = self.lpra_manager.record_task_start(input_task)
            
            # Run original cycle
            result = self.original_graph.cycle(input_task)
            
            # Record results in LPRA
            gen_node_id = self.lpra_manager.record_generation_result(
                task_node_id, result.get("generation", {})
            )
            
            reflect_node_id = self.lpra_manager.record_reflection_result(
                gen_node_id, result.get("reflection")
            )
            
            test_node_id = self.lpra_manager.record_test_result(
                gen_node_id, result.get("test", {})
            )
            
            # Add node IDs to results for reference
            if "generation" in result:
                result["generation"]["node_id"] = gen_node_id
            if "reflection" in result and result["reflection"]:
                result["reflection"]["node_id"] = reflect_node_id
            if "test" in result:
                result["test"]["node_id"] = test_node_id
            
            # Record learning outcome
            learn_node_id = self.lpra_manager.record_learning_outcome(result)
            if "learn" in result:
                result["learn"]["node_id"] = learn_node_id
            
            # Add LPRA context to result
            result["lpra_context"] = self.lpra_manager.get_llm_context(
                focus_nodes=[task_node_id, gen_node_id, test_node_id]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LPRA-integrated cycle failed: {e}")
            # Fallback to original graph
            return self.original_graph.cycle(input_task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including LPRA data."""
        original_stats = self.original_graph.__dict__.copy()
        lpra_stats = self.lpra_manager.get_system_statistics()
        
        return {
            "original_graph": original_stats,
            "lpra": lpra_stats,
            "integration": {
                "lpra_enabled": True,
                "background_maintenance": self.lpra_manager._maintenance_thread.is_alive() if self.lpra_manager._maintenance_thread else False
            }
        }
    
    def get_human_summary(self) -> str:
        """Get human-readable summary of the system state."""
        return self.lpra_manager.get_human_summary()
    
    def cleanup(self) -> None:
        """Cleanup both original graph and LPRA resources."""
        try:
            if hasattr(self.original_graph, 'cleanup'):
                self.original_graph.cleanup()
            self.lpra_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during integrated graph cleanup: {e}")