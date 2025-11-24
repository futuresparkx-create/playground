"""
Semantic Graph Layer (Layer 1) - LPRA Implementation

This module implements the foundational semantic graph layer that models knowledge
as a rich network of typed nodes and relationships with dynamic management mechanisms.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import json
import math

from utils.logging_config import get_logger

logger = get_logger(__name__)


class NodeType(Enum):
    """Hierarchical node type classification."""
    # Agents
    AI_AGENT = "ai_agent"
    HUMAN_OPERATOR = "human_operator"
    EXTERNAL_SYSTEM = "external_system"
    
    # Artifacts
    CODE_FILE = "code_file"
    FUNCTION = "function"
    CLASS = "class"
    TEST_CASE = "test_case"
    
    # Concepts
    PATTERN = "pattern"
    PRINCIPLE = "principle"
    STRATEGY = "strategy"
    
    # Tasks
    GOAL = "goal"
    OBJECTIVE = "objective"
    MILESTONE = "milestone"
    
    # Events
    ACTION = "action"
    DECISION = "decision"
    OUTCOME = "outcome"


class EdgeType(Enum):
    """Typed relationships between nodes."""
    CAUSAL = "causal"              # A causes B
    TEMPORAL = "temporal"          # A happens before/after B
    SEMANTIC = "semantic"          # A is similar/related to B
    DEPENDENCY = "dependency"      # A depends on B
    OWNERSHIP = "ownership"        # A owns/is responsible for B
    IMPROVEMENT = "improvement"    # A improves/enhances B
    VALIDATION = "validation"      # A validates/tests B
    COMPOSITION = "composition"    # A is part of B
    INHERITANCE = "inheritance"    # A inherits from B
    ASSOCIATION = "association"    # A is associated with B


@dataclass
class NodeMetadata:
    """Metadata associated with semantic graph nodes."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    strength: float = 1.0  # Reinforcement strength
    tags: Set[str] = field(default_factory=set)
    source: Optional[str] = None  # Origin of the node
    confidence: float = 1.0  # Confidence in the node's validity


@dataclass
class SemanticNode:
    """A node in the semantic graph representing a concept, entity, or artifact."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType = NodeType.PATTERN
    name: str = ""
    description: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    
    def __post_init__(self):
        """Ensure ID is string and metadata is properly initialized."""
        if not isinstance(self.id, str):
            self.id = str(self.id)
        if not isinstance(self.metadata, NodeMetadata):
            self.metadata = NodeMetadata()
    
    def access(self) -> None:
        """Record access to this node for reinforcement learning."""
        self.metadata.access_count += 1
        self.metadata.last_accessed = datetime.now()
        self.metadata.updated_at = datetime.now()
    
    def reinforce(self, factor: float = 1.1) -> None:
        """Reinforce this node's strength."""
        self.metadata.strength = min(2.0, self.metadata.strength * factor)
        self.metadata.updated_at = datetime.now()
    
    def decay(self, factor: float = 0.95) -> None:
        """Apply time-based decay to this node's strength."""
        self.metadata.strength = max(0.1, self.metadata.strength * factor)
        self.metadata.updated_at = datetime.now()


@dataclass
class EdgeMetadata:
    """Metadata associated with semantic graph edges."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strength: float = 1.0
    confidence: float = 1.0
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class SemanticEdge:
    """A typed relationship between two nodes in the semantic graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    type: EdgeType = EdgeType.ASSOCIATION
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: EdgeMetadata = field(default_factory=EdgeMetadata)
    
    def __post_init__(self):
        """Ensure proper initialization."""
        if not isinstance(self.metadata, EdgeMetadata):
            self.metadata = EdgeMetadata()
    
    def use(self) -> None:
        """Record usage of this edge."""
        self.metadata.usage_count += 1
        self.metadata.last_used = datetime.now()
        self.metadata.updated_at = datetime.now()
    
    def reinforce(self, factor: float = 1.1) -> None:
        """Reinforce this edge's strength."""
        self.metadata.strength = min(2.0, self.metadata.strength * factor)
        self.metadata.updated_at = datetime.now()
    
    def decay(self, factor: float = 0.95) -> None:
        """Apply decay to this edge's strength."""
        self.metadata.strength = max(0.1, self.metadata.strength * factor)
        self.metadata.updated_at = datetime.now()


class GraphMechanism(ABC):
    """Abstract base class for graph management mechanisms."""
    
    @abstractmethod
    def apply(self, graph: 'SemanticGraph') -> None:
        """Apply this mechanism to the graph."""
        pass


class TimeWeightedDecay(GraphMechanism):
    """Applies time-weighted decay to nodes and edges."""
    
    def __init__(self, base_decay_rate: float = 0.95, time_factor: float = 0.001):
        self.base_decay_rate = base_decay_rate
        self.time_factor = time_factor
    
    def apply(self, graph: 'SemanticGraph') -> None:
        """Apply time-weighted decay to all nodes and edges."""
        current_time = datetime.now()
        
        # Decay nodes
        for node in graph.nodes.values():
            if node.metadata.last_accessed:
                time_diff = (current_time - node.metadata.last_accessed).total_seconds()
                decay_factor = self.base_decay_rate * math.exp(-self.time_factor * time_diff)
                node.decay(decay_factor)
        
        # Decay edges
        for edge in graph.edges.values():
            if edge.metadata.last_used:
                time_diff = (current_time - edge.metadata.last_used).total_seconds()
                decay_factor = self.base_decay_rate * math.exp(-self.time_factor * time_diff)
                edge.decay(decay_factor)
        
        logger.debug(f"Applied time-weighted decay to {len(graph.nodes)} nodes and {len(graph.edges)} edges")


class ReinforcementLearning(GraphMechanism):
    """Reinforces successful patterns and relationships."""
    
    def __init__(self, success_boost: float = 1.2, usage_boost: float = 1.1):
        self.success_boost = success_boost
        self.usage_boost = usage_boost
    
    def apply(self, graph: 'SemanticGraph') -> None:
        """Apply reinforcement learning to frequently used elements."""
        # Reinforce frequently accessed nodes
        for node in graph.nodes.values():
            if node.metadata.access_count > 10:  # Threshold for frequent access
                node.reinforce(self.usage_boost)
        
        # Reinforce frequently used edges
        for edge in graph.edges.values():
            if edge.metadata.usage_count > 5:  # Threshold for frequent usage
                edge.reinforce(self.usage_boost)
        
        logger.debug("Applied reinforcement learning to frequently used elements")


class PrincipledPruning(GraphMechanism):
    """Removes low-value nodes and edges based on strength thresholds."""
    
    def __init__(self, node_threshold: float = 0.15, edge_threshold: float = 0.2, max_pruning: int = 100):
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.max_pruning = max_pruning
    
    def apply(self, graph: 'SemanticGraph') -> None:
        """Prune low-strength nodes and edges."""
        pruned_nodes = 0
        pruned_edges = 0
        
        # Prune weak edges first
        edges_to_remove = []
        for edge_id, edge in graph.edges.items():
            if edge.metadata.strength < self.edge_threshold and pruned_edges < self.max_pruning:
                edges_to_remove.append(edge_id)
                pruned_edges += 1
        
        for edge_id in edges_to_remove:
            graph.remove_edge(edge_id)
        
        # Prune weak nodes (but preserve important types)
        nodes_to_remove = []
        for node_id, node in graph.nodes.items():
            if (node.metadata.strength < self.node_threshold and 
                node.type not in [NodeType.GOAL, NodeType.HUMAN_OPERATOR] and
                pruned_nodes < self.max_pruning - pruned_edges):
                nodes_to_remove.append(node_id)
                pruned_nodes += 1
        
        for node_id in nodes_to_remove:
            graph.remove_node(node_id)
        
        logger.info(f"Pruned {pruned_nodes} nodes and {pruned_edges} edges")


class SemanticClustering(GraphMechanism):
    """Groups related concepts and identifies patterns."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def apply(self, graph: 'SemanticGraph') -> None:
        """Identify and create clusters of related nodes."""
        # Simple clustering based on shared edges and similar types
        clusters = {}
        
        for node in graph.nodes.values():
            cluster_key = f"{node.type.value}_{len(graph.get_node_edges(node.id))}"
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(node.id)
        
        # Create cluster nodes for large clusters
        for cluster_key, node_ids in clusters.items():
            if len(node_ids) > 5:  # Threshold for cluster creation
                cluster_node = SemanticNode(
                    type=NodeType.PATTERN,
                    name=f"Cluster_{cluster_key}",
                    description=f"Semantic cluster containing {len(node_ids)} related nodes",
                    content={"member_ids": node_ids}
                )
                graph.add_node(cluster_node)
                
                # Connect cluster to its members
                for node_id in node_ids:
                    edge = SemanticEdge(
                        source_id=cluster_node.id,
                        target_id=node_id,
                        type=EdgeType.COMPOSITION,
                        weight=0.8
                    )
                    graph.add_edge(edge)
        
        logger.debug(f"Created clusters from {len(clusters)} groups")


class SemanticGraph:
    """
    The core semantic graph that manages nodes, edges, and their relationships.
    Implements Layer 1 of the LPRA architecture.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the semantic graph with optional configuration."""
        self.nodes: Dict[str, SemanticNode] = {}
        self.edges: Dict[str, SemanticEdge] = {}
        self.mechanisms: List[GraphMechanism] = []
        self.config = config or {}
        
        # Initialize default mechanisms
        self._initialize_mechanisms()
        
        logger.info("SemanticGraph initialized")
    
    def _initialize_mechanisms(self) -> None:
        """Initialize default graph management mechanisms."""
        self.mechanisms = [
            TimeWeightedDecay(
                base_decay_rate=self.config.get('base_decay_rate', 0.95),
                time_factor=self.config.get('time_decay_factor', 0.001)
            ),
            ReinforcementLearning(
                success_boost=self.config.get('success_boost', 1.2),
                usage_boost=self.config.get('usage_reinforcement', 1.1)
            ),
            PrincipledPruning(
                node_threshold=self.config.get('node_relevance_threshold', 0.15),
                edge_threshold=self.config.get('edge_strength_threshold', 0.2),
                max_pruning=self.config.get('max_pruning_per_cycle', 100)
            ),
            SemanticClustering(
                similarity_threshold=self.config.get('clustering_threshold', 0.7)
            )
        ]
    
    def add_node(self, node: SemanticNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        logger.debug(f"Added node {node.id} of type {node.type.value}")
        return node.id
    
    def add_edge(self, edge: SemanticEdge) -> str:
        """Add an edge to the graph."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"Edge references non-existent nodes: {edge.source_id} -> {edge.target_id}")
        
        self.edges[edge.id] = edge
        logger.debug(f"Added edge {edge.id} of type {edge.type.value}")
        return edge.id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges from the graph."""
        if node_id not in self.nodes:
            return False
        
        # Remove all edges connected to this node
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
        
        del self.nodes[node_id]
        logger.debug(f"Removed node {node_id} and {len(edges_to_remove)} connected edges")
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        if edge_id not in self.edges:
            return False
        
        del self.edges[edge_id]
        logger.debug(f"Removed edge {edge_id}")
        return True
    
    def get_node(self, node_id: str) -> Optional[SemanticNode]:
        """Get a node by ID and record access."""
        node = self.nodes.get(node_id)
        if node:
            node.access()
        return node
    
    def get_edge(self, edge_id: str) -> Optional[SemanticEdge]:
        """Get an edge by ID and record usage."""
        edge = self.edges.get(edge_id)
        if edge:
            edge.use()
        return edge
    
    def get_node_edges(self, node_id: str) -> List[SemanticEdge]:
        """Get all edges connected to a node."""
        return [edge for edge in self.edges.values() 
                if edge.source_id == node_id or edge.target_id == node_id]
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[SemanticNode]:
        """Find all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.type == node_type]
    
    def find_edges_by_type(self, edge_type: EdgeType) -> List[SemanticEdge]:
        """Find all edges of a specific type."""
        return [edge for edge in self.edges.values() if edge.type == edge_type]
    
    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[SemanticNode]:
        """Get neighboring nodes connected by specific edge types."""
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                if not edge_type or edge.type == edge_type:
                    neighbor = self.nodes.get(edge.target_id)
                    if neighbor:
                        neighbors.append(neighbor)
            elif edge.target_id == node_id:
                if not edge_type or edge.type == edge_type:
                    neighbor = self.nodes.get(edge.source_id)
                    if neighbor:
                        neighbors.append(neighbor)
        return neighbors
    
    def apply_mechanisms(self) -> None:
        """Apply all registered graph management mechanisms."""
        logger.info("Applying graph management mechanisms")
        for mechanism in self.mechanisms:
            try:
                mechanism.apply(self)
            except Exception as e:
                logger.error(f"Error applying mechanism {type(mechanism).__name__}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics for monitoring and analysis."""
        node_types = {}
        edge_types = {}
        
        for node in self.nodes.values():
            node_types[node.type.value] = node_types.get(node.type.value, 0) + 1
        
        for edge in self.edges.values():
            edge_types[edge.type.value] = edge_types.get(edge.type.value, 0) + 1
        
        total_strength = sum(node.metadata.strength for node in self.nodes.values())
        avg_strength = total_strength / len(self.nodes) if self.nodes else 0
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "average_node_strength": avg_strength,
            "mechanisms_count": len(self.mechanisms)
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the graph to a dictionary for serialization."""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "type": node.type.value,
                    "name": node.name,
                    "description": node.description,
                    "content": node.content,
                    "metadata": {
                        "created_at": node.metadata.created_at.isoformat(),
                        "updated_at": node.metadata.updated_at.isoformat(),
                        "access_count": node.metadata.access_count,
                        "last_accessed": node.metadata.last_accessed.isoformat() if node.metadata.last_accessed else None,
                        "strength": node.metadata.strength,
                        "tags": list(node.metadata.tags),
                        "source": node.metadata.source,
                        "confidence": node.metadata.confidence
                    }
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                edge_id: {
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "type": edge.type.value,
                    "weight": edge.weight,
                    "properties": edge.properties,
                    "metadata": {
                        "created_at": edge.metadata.created_at.isoformat(),
                        "updated_at": edge.metadata.updated_at.isoformat(),
                        "strength": edge.metadata.strength,
                        "confidence": edge.metadata.confidence,
                        "usage_count": edge.metadata.usage_count,
                        "last_used": edge.metadata.last_used.isoformat() if edge.metadata.last_used else None
                    }
                }
                for edge_id, edge in self.edges.items()
            },
            "config": self.config,
            "exported_at": datetime.now().isoformat()
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save the graph to a JSON file."""
        data = self.export_to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved semantic graph to {filepath}")
    
    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> 'SemanticGraph':
        """Load a graph from a dictionary."""
        graph = cls(config=data.get('config', {}))
        
        # Load nodes
        for node_data in data.get('nodes', {}).values():
            metadata = NodeMetadata(
                created_at=datetime.fromisoformat(node_data['metadata']['created_at']),
                updated_at=datetime.fromisoformat(node_data['metadata']['updated_at']),
                access_count=node_data['metadata']['access_count'],
                last_accessed=datetime.fromisoformat(node_data['metadata']['last_accessed']) if node_data['metadata']['last_accessed'] else None,
                strength=node_data['metadata']['strength'],
                tags=set(node_data['metadata']['tags']),
                source=node_data['metadata']['source'],
                confidence=node_data['metadata']['confidence']
            )
            
            node = SemanticNode(
                id=node_data['id'],
                type=NodeType(node_data['type']),
                name=node_data['name'],
                description=node_data['description'],
                content=node_data['content'],
                metadata=metadata
            )
            graph.add_node(node)
        
        # Load edges
        for edge_data in data.get('edges', {}).values():
            metadata = EdgeMetadata(
                created_at=datetime.fromisoformat(edge_data['metadata']['created_at']),
                updated_at=datetime.fromisoformat(edge_data['metadata']['updated_at']),
                strength=edge_data['metadata']['strength'],
                confidence=edge_data['metadata']['confidence'],
                usage_count=edge_data['metadata']['usage_count'],
                last_used=datetime.fromisoformat(edge_data['metadata']['last_used']) if edge_data['metadata']['last_used'] else None
            )
            
            edge = SemanticEdge(
                id=edge_data['id'],
                source_id=edge_data['source_id'],
                target_id=edge_data['target_id'],
                type=EdgeType(edge_data['type']),
                weight=edge_data['weight'],
                properties=edge_data['properties'],
                metadata=metadata
            )
            graph.add_edge(edge)
        
        return graph
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'SemanticGraph':
        """Load a graph from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.load_from_dict(data)