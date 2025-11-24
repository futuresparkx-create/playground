"""
Surface Context Layer (Layer 3) - LPRA Implementation

This module implements the human and LLM-facing layer that provides compressed,
relevant views of the system state. It handles dynamic context compression,
human-readable summaries, and LLM-optimized context windows.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import math

from memory.semantic_graph import SemanticGraph, SemanticNode, SemanticEdge, NodeType, EdgeType
from memory.structured_state import CanonicalStateSchema, StructuredStateLayer
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ContextType(Enum):
    """Types of context views."""
    HUMAN_SUMMARY = "human_summary"
    LLM_CONTEXT = "llm_context"
    TECHNICAL_REPORT = "technical_report"
    PROGRESS_DASHBOARD = "progress_dashboard"
    DEBUG_VIEW = "debug_view"


class CompressionStrategy(Enum):
    """Context compression strategies."""
    RELEVANCE_BASED = "relevance_based"
    TEMPORAL_SLIDING = "temporal_sliding"
    HIERARCHICAL = "hierarchical"
    ATTENTION_GUIDED = "attention_guided"
    TOKEN_OPTIMIZED = "token_optimized"


@dataclass
class ContextWindow:
    """Represents a context window with metadata."""
    content: str
    context_type: ContextType
    compression_strategy: CompressionStrategy
    token_count: int
    relevance_score: float
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionWeights:
    """Attention weights for different graph elements."""
    node_weights: Dict[str, float] = field(default_factory=dict)
    edge_weights: Dict[str, float] = field(default_factory=dict)
    type_weights: Dict[str, float] = field(default_factory=dict)
    temporal_decay: float = 0.95
    focus_threshold: float = 0.7


class ContextCompressor(ABC):
    """Abstract base class for context compression strategies."""
    
    @abstractmethod
    def compress(self, 
                 graph: SemanticGraph, 
                 state: CanonicalStateSchema,
                 attention: AttentionWeights,
                 max_tokens: int = 4000) -> ContextWindow:
        """Compress the graph and state into a context window."""
        pass


class RelevanceBasedCompressor(ContextCompressor):
    """Compresses context based on relevance scoring."""
    
    def __init__(self, relevance_threshold: float = 0.5):
        self.relevance_threshold = relevance_threshold
    
    def compress(self, 
                 graph: SemanticGraph, 
                 state: CanonicalStateSchema,
                 attention: AttentionWeights,
                 max_tokens: int = 4000) -> ContextWindow:
        """Compress based on relevance scores."""
        
        # Score nodes by relevance
        relevant_nodes = []
        for node in graph.nodes.values():
            relevance = self._compute_node_relevance(node, attention)
            if relevance >= self.relevance_threshold:
                relevant_nodes.append((node, relevance))
        
        # Sort by relevance
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Build compressed content
        content_parts = []
        token_count = 0
        
        # Add high-level summary
        summary = f"Graph State: {state.node_count} nodes, {state.edge_count} edges\n"
        summary += f"Integrity: {state.integrity_score:.2f}, Strength: {state.average_node_strength:.2f}\n\n"
        content_parts.append(summary)
        token_count += len(summary.split())
        
        # Add relevant nodes
        for node, relevance in relevant_nodes:
            if token_count >= max_tokens * 0.8:  # Reserve 20% for edges
                break
            
            node_desc = f"[{node.type.value}] {node.name}: {node.description[:100]}...\n"
            node_tokens = len(node_desc.split())
            
            if token_count + node_tokens <= max_tokens * 0.8:
                content_parts.append(node_desc)
                token_count += node_tokens
        
        # Add relevant edges
        relevant_edges = self._get_relevant_edges(graph, [n[0] for n in relevant_nodes], attention)
        for edge in relevant_edges[:10]:  # Limit edges
            if token_count >= max_tokens:
                break
            
            source_name = graph.nodes[edge.source_id].name
            target_name = graph.nodes[edge.target_id].name
            edge_desc = f"{source_name} --[{edge.type.value}]--> {target_name}\n"
            edge_tokens = len(edge_desc.split())
            
            if token_count + edge_tokens <= max_tokens:
                content_parts.append(edge_desc)
                token_count += edge_tokens
        
        content = "".join(content_parts)
        
        return ContextWindow(
            content=content,
            context_type=ContextType.LLM_CONTEXT,
            compression_strategy=CompressionStrategy.RELEVANCE_BASED,
            token_count=token_count,
            relevance_score=sum(r for _, r in relevant_nodes) / len(relevant_nodes) if relevant_nodes else 0.0,
            metadata={"nodes_included": len(relevant_nodes), "edges_included": len(relevant_edges)}
        )
    
    def _compute_node_relevance(self, node: SemanticNode, attention: AttentionWeights) -> float:
        """Compute relevance score for a node."""
        base_score = node.metadata.strength
        
        # Apply attention weights
        if node.id in attention.node_weights:
            base_score *= attention.node_weights[node.id]
        
        # Type-based weighting
        if node.type.value in attention.type_weights:
            base_score *= attention.type_weights[node.type.value]
        
        # Temporal decay
        age_hours = (datetime.now() - node.metadata.created_at).total_seconds() / 3600
        temporal_factor = attention.temporal_decay ** age_hours
        
        # Access frequency boost
        access_boost = min(2.0, 1.0 + node.metadata.access_count * 0.1)
        
        return base_score * temporal_factor * access_boost
    
    def _get_relevant_edges(self, 
                           graph: SemanticGraph, 
                           relevant_nodes: List[SemanticNode],
                           attention: AttentionWeights) -> List[SemanticEdge]:
        """Get edges relevant to the selected nodes."""
        relevant_node_ids = {node.id for node in relevant_nodes}
        relevant_edges = []
        
        for edge in graph.edges.values():
            if edge.source_id in relevant_node_ids or edge.target_id in relevant_node_ids:
                relevance = edge.metadata.strength
                if edge.id in attention.edge_weights:
                    relevance *= attention.edge_weights[edge.id]
                relevant_edges.append((edge, relevance))
        
        # Sort by relevance
        relevant_edges.sort(key=lambda x: x[1], reverse=True)
        return [edge for edge, _ in relevant_edges]


class TemporalSlidingCompressor(ContextCompressor):
    """Compresses context using a sliding temporal window."""
    
    def __init__(self, window_hours: float = 24.0):
        self.window_hours = window_hours
    
    def compress(self, 
                 graph: SemanticGraph, 
                 state: CanonicalStateSchema,
                 attention: AttentionWeights,
                 max_tokens: int = 4000) -> ContextWindow:
        """Compress using temporal sliding window."""
        
        cutoff_time = datetime.now() - timedelta(hours=self.window_hours)
        
        # Filter nodes by time window
        recent_nodes = [
            node for node in graph.nodes.values()
            if node.metadata.updated_at >= cutoff_time
        ]
        
        # Sort by recency
        recent_nodes.sort(key=lambda x: x.metadata.updated_at, reverse=True)
        
        content_parts = []
        token_count = 0
        
        # Add temporal summary
        summary = f"Recent Activity (last {self.window_hours}h): {len(recent_nodes)} active nodes\n"
        summary += f"System State: {state.node_count} total nodes, integrity {state.integrity_score:.2f}\n\n"
        content_parts.append(summary)
        token_count += len(summary.split())
        
        # Add recent nodes chronologically
        for node in recent_nodes:
            if token_count >= max_tokens * 0.9:
                break
            
            time_str = node.metadata.updated_at.strftime("%H:%M")
            node_desc = f"[{time_str}] {node.type.value}: {node.name}\n"
            if node.description:
                node_desc += f"  {node.description[:80]}...\n"
            
            node_tokens = len(node_desc.split())
            if token_count + node_tokens <= max_tokens * 0.9:
                content_parts.append(node_desc)
                token_count += node_tokens
        
        content = "".join(content_parts)
        
        return ContextWindow(
            content=content,
            context_type=ContextType.LLM_CONTEXT,
            compression_strategy=CompressionStrategy.TEMPORAL_SLIDING,
            token_count=token_count,
            relevance_score=0.8,  # Temporal relevance
            metadata={"window_hours": self.window_hours, "recent_nodes": len(recent_nodes)}
        )


class HierarchicalCompressor(ContextCompressor):
    """Compresses context using hierarchical summarization."""
    
    def compress(self, 
                 graph: SemanticGraph, 
                 state: CanonicalStateSchema,
                 attention: AttentionWeights,
                 max_tokens: int = 4000) -> ContextWindow:
        """Compress using hierarchical structure."""
        
        content_parts = []
        token_count = 0
        
        # Level 1: System overview
        overview = f"=== SYSTEM OVERVIEW ===\n"
        overview += f"Nodes: {state.node_count}, Edges: {state.edge_count}\n"
        overview += f"Integrity: {state.integrity_score:.2f}, Avg Strength: {state.average_node_strength:.2f}\n"
        overview += f"Clusters: {state.cluster_count}, Hot Nodes: {len(state.hot_nodes)}\n\n"
        content_parts.append(overview)
        token_count += len(overview.split())
        
        # Level 2: Type summaries
        type_summary = "=== TYPE DISTRIBUTION ===\n"
        for node_type, count in state.node_type_counts.items():
            type_summary += f"{node_type}: {count} nodes\n"
        type_summary += "\n"
        content_parts.append(type_summary)
        token_count += len(type_summary.split())
        
        # Level 3: Key nodes by type
        key_nodes_section = "=== KEY NODES ===\n"
        for node_type in NodeType:
            nodes_of_type = graph.find_nodes_by_type(node_type)
            if nodes_of_type and token_count < max_tokens * 0.7:
                # Get strongest nodes of this type
                strongest_nodes = sorted(nodes_of_type, 
                                       key=lambda x: x.metadata.strength, 
                                       reverse=True)[:3]
                
                key_nodes_section += f"\n{node_type.value.upper()}:\n"
                for node in strongest_nodes:
                    if token_count >= max_tokens * 0.7:
                        break
                    node_line = f"  • {node.name} (strength: {node.metadata.strength:.2f})\n"
                    node_tokens = len(node_line.split())
                    if token_count + node_tokens <= max_tokens * 0.7:
                        key_nodes_section += node_line
                        token_count += node_tokens
        
        content_parts.append(key_nodes_section)
        
        # Level 4: Recent activity (if space allows)
        if token_count < max_tokens * 0.8:
            recent_activity = "\n=== RECENT ACTIVITY ===\n"
            recent_nodes = sorted(graph.nodes.values(), 
                                key=lambda x: x.metadata.updated_at, 
                                reverse=True)[:5]
            
            for node in recent_nodes:
                if token_count >= max_tokens:
                    break
                activity_line = f"• {node.name} ({node.metadata.updated_at.strftime('%H:%M')})\n"
                activity_tokens = len(activity_line.split())
                if token_count + activity_tokens <= max_tokens:
                    recent_activity += activity_line
                    token_count += activity_tokens
            
            content_parts.append(recent_activity)
        
        content = "".join(content_parts)
        
        return ContextWindow(
            content=content,
            context_type=ContextType.TECHNICAL_REPORT,
            compression_strategy=CompressionStrategy.HIERARCHICAL,
            token_count=token_count,
            relevance_score=0.9,  # High relevance for structured view
            metadata={"levels_included": 4, "type_coverage": len(state.node_type_counts)}
        )


class HumanSummaryGenerator:
    """Generates human-readable summaries and progress reports."""
    
    def generate_progress_summary(self, 
                                 graph: SemanticGraph, 
                                 state: CanonicalStateSchema) -> str:
        """Generate a human-readable progress summary."""
        
        summary_parts = []
        
        # Header
        summary_parts.append("🚀 AI Code Improvement Playground - Progress Report")
        summary_parts.append("=" * 60)
        summary_parts.append("")
        
        # System health
        health_emoji = "🟢" if state.integrity_score > 0.8 else "🟡" if state.integrity_score > 0.6 else "🔴"
        summary_parts.append(f"{health_emoji} System Health: {state.integrity_score:.1%}")
        summary_parts.append(f"📊 Knowledge Base: {state.node_count} concepts, {state.edge_count} relationships")
        summary_parts.append(f"💪 Average Strength: {state.average_node_strength:.2f}/2.0")
        summary_parts.append("")
        
        # Activity overview
        summary_parts.append("📈 Activity Overview:")
        if state.hot_nodes:
            summary_parts.append(f"  🔥 {len(state.hot_nodes)} highly active concepts")
        if state.cold_nodes:
            summary_parts.append(f"  ❄️  {len(state.cold_nodes)} unused concepts (candidates for cleanup)")
        summary_parts.append(f"  🎯 {state.cluster_count} knowledge clusters identified")
        summary_parts.append("")
        
        # Type breakdown
        summary_parts.append("🏗️ Knowledge Composition:")
        for node_type, count in sorted(state.node_type_counts.items()):
            emoji = self._get_type_emoji(node_type)
            summary_parts.append(f"  {emoji} {node_type.replace('_', ' ').title()}: {count}")
        summary_parts.append("")
        
        # Recent achievements
        goals = graph.find_nodes_by_type(NodeType.GOAL)
        completed_goals = [g for g in goals if g.metadata.strength > 1.5]
        if completed_goals:
            summary_parts.append("🎉 Recent Achievements:")
            for goal in completed_goals[:3]:
                summary_parts.append(f"  ✅ {goal.name}")
            summary_parts.append("")
        
        # Recommendations
        summary_parts.append("💡 Recommendations:")
        if state.orphaned_edges > 0:
            summary_parts.append(f"  🔧 Clean up {state.orphaned_edges} orphaned relationships")
        if state.circular_dependencies > 0:
            summary_parts.append(f"  ⚠️  Resolve {state.circular_dependencies} circular dependencies")
        if len(state.cold_nodes) > 10:
            summary_parts.append(f"  🧹 Consider pruning {len(state.cold_nodes)} unused concepts")
        if not summary_parts[-1].startswith("  "):
            summary_parts.append("  🎯 System is running optimally!")
        summary_parts.append("")
        
        # Footer
        summary_parts.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append("=" * 60)
        
        return "\n".join(summary_parts)
    
    def generate_debug_report(self, 
                             graph: SemanticGraph, 
                             state: CanonicalStateSchema) -> str:
        """Generate a technical debug report."""
        
        report_parts = []
        
        # Header
        report_parts.append("🔍 LPRA Debug Report")
        report_parts.append("=" * 50)
        report_parts.append("")
        
        # Graph statistics
        report_parts.append("📊 Graph Statistics:")
        report_parts.append(f"  Nodes: {len(graph.nodes)}")
        report_parts.append(f"  Edges: {len(graph.edges)}")
        report_parts.append(f"  Density: {len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1)) * 2:.4f}")
        report_parts.append("")
        
        # Integrity analysis
        report_parts.append("🔍 Integrity Analysis:")
        report_parts.append(f"  Overall Score: {state.integrity_score:.4f}")
        report_parts.append(f"  Orphaned Edges: {state.orphaned_edges}")
        report_parts.append(f"  Circular Dependencies: {state.circular_dependencies}")
        report_parts.append("")
        
        # Strength distribution
        node_strengths = [node.metadata.strength for node in graph.nodes.values()]
        if node_strengths:
            report_parts.append("💪 Strength Distribution:")
            report_parts.append(f"  Min: {min(node_strengths):.3f}")
            report_parts.append(f"  Max: {max(node_strengths):.3f}")
            report_parts.append(f"  Mean: {sum(node_strengths) / len(node_strengths):.3f}")
            report_parts.append(f"  Std: {self._compute_std(node_strengths):.3f}")
            report_parts.append("")
        
        # Access patterns
        report_parts.append("🎯 Access Patterns:")
        total_accesses = sum(node.metadata.access_count for node in graph.nodes.values())
        report_parts.append(f"  Total Accesses: {total_accesses}")
        if total_accesses > 0:
            most_accessed = max(graph.nodes.values(), key=lambda x: x.metadata.access_count)
            report_parts.append(f"  Most Accessed: {most_accessed.name} ({most_accessed.metadata.access_count})")
        report_parts.append("")
        
        # Memory usage estimation
        report_parts.append("💾 Memory Estimation:")
        node_memory = len(graph.nodes) * 1024  # Rough estimate
        edge_memory = len(graph.edges) * 512   # Rough estimate
        total_memory = node_memory + edge_memory
        report_parts.append(f"  Nodes: ~{node_memory / 1024:.1f} KB")
        report_parts.append(f"  Edges: ~{edge_memory / 1024:.1f} KB")
        report_parts.append(f"  Total: ~{total_memory / 1024:.1f} KB")
        report_parts.append("")
        
        # Timestamp
        report_parts.append(f"Generated: {datetime.now().isoformat()}")
        
        return "\n".join(report_parts)
    
    def _get_type_emoji(self, node_type: str) -> str:
        """Get emoji for node type."""
        emoji_map = {
            "ai_agent": "🤖",
            "human_operator": "👤",
            "external_system": "🔌",
            "code_file": "📄",
            "function": "⚙️",
            "class": "🏗️",
            "test_case": "🧪",
            "pattern": "🔄",
            "principle": "📏",
            "strategy": "🎯",
            "goal": "🎯",
            "objective": "📋",
            "milestone": "🏁",
            "action": "⚡",
            "decision": "🤔",
            "outcome": "📊"
        }
        return emoji_map.get(node_type, "📦")
    
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


class SurfaceContextLayer:
    """
    Layer 3 of LPRA - Provides human and LLM-facing context views.
    Handles dynamic compression, summarization, and context window management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the surface context layer."""
        self.config = config or {}
        
        # Initialize compressors
        self.compressors = {
            CompressionStrategy.RELEVANCE_BASED: RelevanceBasedCompressor(
                relevance_threshold=self.config.get('relevance_threshold', 0.5)
            ),
            CompressionStrategy.TEMPORAL_SLIDING: TemporalSlidingCompressor(
                window_hours=self.config.get('temporal_window_hours', 24.0)
            ),
            CompressionStrategy.HIERARCHICAL: HierarchicalCompressor()
        }
        
        self.summary_generator = HumanSummaryGenerator()
        
        # Context cache
        self.context_cache: Dict[str, ContextWindow] = {}
        self.cache_ttl_minutes = self.config.get('cache_ttl_minutes', 30)
        
        logger.info("SurfaceContextLayer initialized")
    
    def get_llm_context(self, 
                       graph: SemanticGraph,
                       state: CanonicalStateSchema,
                       strategy: CompressionStrategy = CompressionStrategy.RELEVANCE_BASED,
                       max_tokens: int = 4000,
                       attention: Optional[AttentionWeights] = None) -> ContextWindow:
        """Get LLM-optimized context window."""
        
        if attention is None:
            attention = self._create_default_attention(graph)
        
        # Check cache
        cache_key = f"llm_{strategy.value}_{max_tokens}_{hash(str(attention.node_weights))}"
        cached_context = self.context_cache.get(cache_key)
        
        if cached_context and self._is_cache_valid(cached_context):
            logger.debug(f"Returning cached LLM context: {cache_key}")
            return cached_context
        
        # Generate new context
        compressor = self.compressors.get(strategy)
        if not compressor:
            logger.warning(f"Unknown compression strategy: {strategy}, using relevance-based")
            compressor = self.compressors[CompressionStrategy.RELEVANCE_BASED]
        
        context = compressor.compress(graph, state, attention, max_tokens)
        
        # Cache the result
        context.expires_at = datetime.now() + timedelta(minutes=self.cache_ttl_minutes)
        self.context_cache[cache_key] = context
        
        logger.info(f"Generated LLM context: {context.token_count} tokens, relevance {context.relevance_score:.3f}")
        return context
    
    def get_human_summary(self, 
                         graph: SemanticGraph,
                         state: CanonicalStateSchema,
                         summary_type: str = "progress") -> str:
        """Get human-readable summary."""
        
        if summary_type == "progress":
            return self.summary_generator.generate_progress_summary(graph, state)
        elif summary_type == "debug":
            return self.summary_generator.generate_debug_report(graph, state)
        else:
            logger.warning(f"Unknown summary type: {summary_type}, using progress")
            return self.summary_generator.generate_progress_summary(graph, state)
    
    def create_attention_weights(self, 
                               focus_nodes: Optional[List[str]] = None,
                               focus_types: Optional[List[NodeType]] = None,
                               temporal_decay: float = 0.95,
                               focus_threshold: float = 0.7) -> AttentionWeights:
        """Create custom attention weights for context generation."""
        
        attention = AttentionWeights(
            temporal_decay=temporal_decay,
            focus_threshold=focus_threshold
        )
        
        # Set node-specific weights
        if focus_nodes:
            for node_id in focus_nodes:
                attention.node_weights[node_id] = 2.0  # High attention
        
        # Set type-specific weights
        if focus_types:
            for node_type in focus_types:
                attention.type_weights[node_type.value] = 1.5  # Elevated attention
        
        return attention
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context generation and caching."""
        
        # Clean expired cache entries
        self._cleanup_cache()
        
        cache_stats = {
            "cached_contexts": len(self.context_cache),
            "cache_hit_types": {},
            "average_token_count": 0,
            "average_relevance_score": 0
        }
        
        if self.context_cache:
            total_tokens = 0
            total_relevance = 0
            
            for context in self.context_cache.values():
                cache_stats["cache_hit_types"][context.compression_strategy.value] = \
                    cache_stats["cache_hit_types"].get(context.compression_strategy.value, 0) + 1
                total_tokens += context.token_count
                total_relevance += context.relevance_score
            
            cache_stats["average_token_count"] = total_tokens / len(self.context_cache)
            cache_stats["average_relevance_score"] = total_relevance / len(self.context_cache)
        
        return cache_stats
    
    def clear_cache(self) -> int:
        """Clear the context cache and return number of cleared entries."""
        cleared_count = len(self.context_cache)
        self.context_cache.clear()
        logger.info(f"Cleared {cleared_count} cached context entries")
        return cleared_count
    
    def _create_default_attention(self, graph: SemanticGraph) -> AttentionWeights:
        """Create default attention weights based on graph characteristics."""
        attention = AttentionWeights()
        
        # Boost attention for frequently accessed nodes
        for node in graph.nodes.values():
            if node.metadata.access_count > 5:
                attention.node_weights[node.id] = 1.0 + min(1.0, node.metadata.access_count * 0.1)
        
        # Boost attention for important types
        important_types = [NodeType.GOAL, NodeType.HUMAN_OPERATOR, NodeType.OUTCOME]
        for node_type in important_types:
            attention.type_weights[node_type.value] = 1.3
        
        return attention
    
    def _is_cache_valid(self, context: ContextWindow) -> bool:
        """Check if a cached context is still valid."""
        if context.expires_at is None:
            return True
        return datetime.now() < context.expires_at
    
    def _cleanup_cache(self) -> None:
        """Remove expired entries from the cache."""
        expired_keys = []
        for key, context in self.context_cache.items():
            if not self._is_cache_valid(context):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.context_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")