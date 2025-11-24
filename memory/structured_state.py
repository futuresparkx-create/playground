"""
Structured State Layer (Layer 2) - LPRA Implementation

This module implements the canonical, machine-readable state representation
derived from the semantic graph. It provides deterministic state computation,
consistency checking, and version management.
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple
import hashlib
import threading
from contextlib import contextmanager

import lancedb
import numpy as np
from pydantic import BaseModel, Field, validator

from memory.semantic_graph import SemanticGraph, SemanticNode, SemanticEdge, NodeType, EdgeType
from utils.logging_config import get_logger
from utils.exceptions import PlaygroundException

logger = get_logger(__name__)


class StateVersion(Enum):
    """State schema versions for migration support."""
    V0_1_BASELINE = "0.1"
    V0_2_HIERARCHICAL = "0.2"
    V0_3_MULTIAGENT = "0.3"
    V1_0_STABLE = "1.0"


class ConsistencyLevel(Enum):
    """Consistency checking levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class StateMetadata:
    """Metadata for state snapshots."""
    version: StateVersion = StateVersion.V1_0_STABLE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    consistency_level: ConsistencyLevel = ConsistencyLevel.STRICT
    source_graph_hash: str = ""
    derivation_time_ms: float = 0.0


class CanonicalStateSchema(BaseModel):
    """Pydantic schema for canonical state validation."""
    
    class Config:
        arbitrary_types_allowed = True
    
    # Core identifiers
    state_id: str = Field(..., description="Unique state identifier")
    version: str = Field(default="1.0", description="Schema version")
    
    # Graph summary
    node_count: int = Field(ge=0, description="Total number of nodes")
    edge_count: int = Field(ge=0, description="Total number of edges")
    
    # Type distributions
    node_type_counts: Dict[str, int] = Field(default_factory=dict)
    edge_type_counts: Dict[str, int] = Field(default_factory=dict)
    
    # Strength metrics
    average_node_strength: float = Field(ge=0.0, le=2.0)
    average_edge_strength: float = Field(ge=0.0, le=2.0)
    total_graph_strength: float = Field(ge=0.0)
    
    # Temporal information
    oldest_node_age_hours: float = Field(ge=0.0)
    newest_node_age_hours: float = Field(ge=0.0)
    last_mechanism_run: Optional[datetime] = None
    
    # Clustering information
    cluster_count: int = Field(ge=0)
    largest_cluster_size: int = Field(ge=0)
    
    # Performance metrics
    access_patterns: Dict[str, int] = Field(default_factory=dict)
    hot_nodes: List[str] = Field(default_factory=list)
    cold_nodes: List[str] = Field(default_factory=list)
    
    # Consistency indicators
    orphaned_edges: int = Field(ge=0)
    circular_dependencies: int = Field(ge=0)
    integrity_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    @validator('node_type_counts')
    def validate_node_types(cls, v):
        """Validate that all node types are recognized."""
        valid_types = {nt.value for nt in NodeType}
        for node_type in v.keys():
            if node_type not in valid_types:
                raise ValueError(f"Unknown node type: {node_type}")
        return v
    
    @validator('edge_type_counts')
    def validate_edge_types(cls, v):
        """Validate that all edge types are recognized."""
        valid_types = {et.value for et in EdgeType}
        for edge_type in v.keys():
            if edge_type not in valid_types:
                raise ValueError(f"Unknown edge type: {edge_type}")
        return v


class StateStore(ABC):
    """Abstract base class for state storage backends."""
    
    @abstractmethod
    def save_state(self, state_id: str, state_data: Dict[str, Any], metadata: StateMetadata) -> bool:
        """Save a state snapshot."""
        pass
    
    @abstractmethod
    def load_state(self, state_id: str) -> Optional[Tuple[Dict[str, Any], StateMetadata]]:
        """Load a state snapshot."""
        pass
    
    @abstractmethod
    def list_states(self, limit: int = 100) -> List[Tuple[str, StateMetadata]]:
        """List available state snapshots."""
        pass
    
    @abstractmethod
    def delete_state(self, state_id: str) -> bool:
        """Delete a state snapshot."""
        pass
    
    @abstractmethod
    def cleanup_old_states(self, keep_count: int = 10) -> int:
        """Clean up old state snapshots, keeping the most recent ones."""
        pass


class SQLiteStateStore(StateStore):
    """SQLite-based state storage implementation."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS states (
                    state_id TEXT PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    consistency_level TEXT NOT NULL,
                    source_graph_hash TEXT NOT NULL,
                    derivation_time_ms REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_states_created_at ON states(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_states_version ON states(version)
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper locking."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def save_state(self, state_id: str, state_data: Dict[str, Any], metadata: StateMetadata) -> bool:
        """Save a state snapshot to SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO states 
                    (state_id, state_data, version, created_at, updated_at, 
                     checksum, consistency_level, source_graph_hash, derivation_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state_id,
                    json.dumps(state_data),
                    metadata.version.value,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    metadata.checksum,
                    metadata.consistency_level.value,
                    metadata.source_graph_hash,
                    metadata.derivation_time_ms
                ))
            logger.debug(f"Saved state {state_id} to SQLite")
            return True
        except Exception as e:
            logger.error(f"Failed to save state {state_id}: {e}")
            return False
    
    def load_state(self, state_id: str) -> Optional[Tuple[Dict[str, Any], StateMetadata]]:
        """Load a state snapshot from SQLite."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM states WHERE state_id = ?", (state_id,)
                ).fetchone()
                
                if not row:
                    return None
                
                state_data = json.loads(row['state_data'])
                metadata = StateMetadata(
                    version=StateVersion(row['version']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    checksum=row['checksum'],
                    consistency_level=ConsistencyLevel(row['consistency_level']),
                    source_graph_hash=row['source_graph_hash'],
                    derivation_time_ms=row['derivation_time_ms']
                )
                
                return state_data, metadata
        except Exception as e:
            logger.error(f"Failed to load state {state_id}: {e}")
            return None
    
    def list_states(self, limit: int = 100) -> List[Tuple[str, StateMetadata]]:
        """List available state snapshots."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT state_id, version, created_at, updated_at, checksum, 
                           consistency_level, source_graph_hash, derivation_time_ms
                    FROM states 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,)).fetchall()
                
                result = []
                for row in rows:
                    metadata = StateMetadata(
                        version=StateVersion(row['version']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        checksum=row['checksum'],
                        consistency_level=ConsistencyLevel(row['consistency_level']),
                        source_graph_hash=row['source_graph_hash'],
                        derivation_time_ms=row['derivation_time_ms']
                    )
                    result.append((row['state_id'], metadata))
                
                return result
        except Exception as e:
            logger.error(f"Failed to list states: {e}")
            return []
    
    def delete_state(self, state_id: str) -> bool:
        """Delete a state snapshot."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM states WHERE state_id = ?", (state_id,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False
    
    def cleanup_old_states(self, keep_count: int = 10) -> int:
        """Clean up old state snapshots."""
        try:
            with self._get_connection() as conn:
                # Get states to delete (all except the most recent keep_count)
                rows = conn.execute("""
                    SELECT state_id FROM states 
                    ORDER BY created_at DESC 
                    LIMIT -1 OFFSET ?
                """, (keep_count,)).fetchall()
                
                deleted_count = 0
                for row in rows:
                    conn.execute("DELETE FROM states WHERE state_id = ?", (row['state_id'],))
                    deleted_count += 1
                
                logger.info(f"Cleaned up {deleted_count} old state snapshots")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old states: {e}")
            return 0


class LanceDBStateStore(StateStore):
    """LanceDB-based state storage for vector embeddings."""
    
    def __init__(self, db_path: Path, table_name: str = "states"):
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(str(db_path))
        self._initialize_table()
    
    def _initialize_table(self) -> None:
        """Initialize the LanceDB table."""
        try:
            # Check if table exists
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Create table with schema
            schema = [
                {"name": "state_id", "type": "string"},
                {"name": "embedding", "type": "vector"},
                {"name": "state_data", "type": "string"},
                {"name": "metadata", "type": "string"},
                {"name": "created_at", "type": "string"}
            ]
            
            # Create with dummy data to establish schema
            dummy_data = [{
                "state_id": "dummy",
                "embedding": np.zeros(384).tolist(),  # Default embedding size
                "state_data": "{}",
                "metadata": "{}",
                "created_at": datetime.now().isoformat()
            }]
            
            self.table = self.db.create_table(self.table_name, dummy_data)
            # Remove dummy data
            self.table.delete("state_id = 'dummy'")
    
    def _compute_state_embedding(self, state_data: Dict[str, Any]) -> np.ndarray:
        """Compute vector embedding for state data."""
        # Simple embedding based on state characteristics
        # In a real implementation, you'd use a proper embedding model
        features = [
            state_data.get('node_count', 0),
            state_data.get('edge_count', 0),
            state_data.get('average_node_strength', 0),
            state_data.get('average_edge_strength', 0),
            state_data.get('cluster_count', 0),
            state_data.get('integrity_score', 1.0)
        ]
        
        # Pad to 384 dimensions (typical embedding size)
        embedding = np.zeros(384)
        embedding[:len(features)] = features
        
        return embedding
    
    def save_state(self, state_id: str, state_data: Dict[str, Any], metadata: StateMetadata) -> bool:
        """Save a state snapshot to LanceDB."""
        try:
            embedding = self._compute_state_embedding(state_data)
            
            data = [{
                "state_id": state_id,
                "embedding": embedding.tolist(),
                "state_data": json.dumps(state_data),
                "metadata": json.dumps(asdict(metadata), default=str),
                "created_at": metadata.created_at.isoformat()
            }]
            
            # Delete existing state if it exists
            try:
                self.table.delete(f"state_id = '{state_id}'")
            except Exception:
                pass  # State might not exist
            
            self.table.add(data)
            logger.debug(f"Saved state {state_id} to LanceDB")
            return True
        except Exception as e:
            logger.error(f"Failed to save state {state_id} to LanceDB: {e}")
            return False
    
    def load_state(self, state_id: str) -> Optional[Tuple[Dict[str, Any], StateMetadata]]:
        """Load a state snapshot from LanceDB."""
        try:
            results = self.table.search().where(f"state_id = '{state_id}'").limit(1).to_list()
            
            if not results:
                return None
            
            result = results[0]
            state_data = json.loads(result['state_data'])
            metadata_dict = json.loads(result['metadata'])
            
            # Reconstruct metadata
            metadata = StateMetadata(
                version=StateVersion(metadata_dict['version']),
                created_at=datetime.fromisoformat(metadata_dict['created_at']),
                updated_at=datetime.fromisoformat(metadata_dict['updated_at']),
                checksum=metadata_dict['checksum'],
                consistency_level=ConsistencyLevel(metadata_dict['consistency_level']),
                source_graph_hash=metadata_dict['source_graph_hash'],
                derivation_time_ms=metadata_dict['derivation_time_ms']
            )
            
            return state_data, metadata
        except Exception as e:
            logger.error(f"Failed to load state {state_id} from LanceDB: {e}")
            return None
    
    def list_states(self, limit: int = 100) -> List[Tuple[str, StateMetadata]]:
        """List available state snapshots."""
        try:
            results = self.table.search().limit(limit).to_list()
            
            state_list = []
            for result in results:
                metadata_dict = json.loads(result['metadata'])
                metadata = StateMetadata(
                    version=StateVersion(metadata_dict['version']),
                    created_at=datetime.fromisoformat(metadata_dict['created_at']),
                    updated_at=datetime.fromisoformat(metadata_dict['updated_at']),
                    checksum=metadata_dict['checksum'],
                    consistency_level=ConsistencyLevel(metadata_dict['consistency_level']),
                    source_graph_hash=metadata_dict['source_graph_hash'],
                    derivation_time_ms=metadata_dict['derivation_time_ms']
                )
                state_list.append((result['state_id'], metadata))
            
            return sorted(state_list, key=lambda x: x[1].created_at, reverse=True)
        except Exception as e:
            logger.error(f"Failed to list states from LanceDB: {e}")
            return []
    
    def delete_state(self, state_id: str) -> bool:
        """Delete a state snapshot."""
        try:
            self.table.delete(f"state_id = '{state_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete state {state_id} from LanceDB: {e}")
            return False
    
    def cleanup_old_states(self, keep_count: int = 10) -> int:
        """Clean up old state snapshots."""
        try:
            states = self.list_states(limit=1000)  # Get more than we need
            if len(states) <= keep_count:
                return 0
            
            # Delete oldest states
            states_to_delete = states[keep_count:]
            deleted_count = 0
            
            for state_id, _ in states_to_delete:
                if self.delete_state(state_id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old state snapshots from LanceDB")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old states from LanceDB: {e}")
            return 0


class DerivationEngine:
    """
    Computes canonical state from semantic graph with consistency checking
    and version management.
    """
    
    def __init__(self, consistency_level: ConsistencyLevel = ConsistencyLevel.STRICT):
        self.consistency_level = consistency_level
        self.version = StateVersion.V1_0_STABLE
    
    def derive_state(self, graph: SemanticGraph) -> CanonicalStateSchema:
        """Derive canonical state from semantic graph."""
        start_time = datetime.now()
        
        try:
            # Compute basic metrics
            node_count = len(graph.nodes)
            edge_count = len(graph.edges)
            
            # Type distributions
            node_type_counts = {}
            for node in graph.nodes.values():
                node_type_counts[node.type.value] = node_type_counts.get(node.type.value, 0) + 1
            
            edge_type_counts = {}
            for edge in graph.edges.values():
                edge_type_counts[edge.type.value] = edge_type_counts.get(edge.type.value, 0) + 1
            
            # Strength metrics
            node_strengths = [node.metadata.strength for node in graph.nodes.values()]
            edge_strengths = [edge.metadata.strength for edge in graph.edges.values()]
            
            avg_node_strength = sum(node_strengths) / len(node_strengths) if node_strengths else 0.0
            avg_edge_strength = sum(edge_strengths) / len(edge_strengths) if edge_strengths else 0.0
            total_strength = sum(node_strengths) + sum(edge_strengths)
            
            # Temporal information
            now = datetime.now()
            node_ages = [(now - node.metadata.created_at).total_seconds() / 3600 for node in graph.nodes.values()]
            oldest_age = max(node_ages) if node_ages else 0.0
            newest_age = min(node_ages) if node_ages else 0.0
            
            # Clustering analysis
            clusters = self._analyze_clusters(graph)
            cluster_count = len(clusters)
            largest_cluster_size = max(len(cluster) for cluster in clusters) if clusters else 0
            
            # Access patterns
            access_patterns = {}
            hot_nodes = []
            cold_nodes = []
            
            for node in graph.nodes.values():
                access_count = node.metadata.access_count
                access_patterns[node.type.value] = access_patterns.get(node.type.value, 0) + access_count
                
                if access_count > 10:
                    hot_nodes.append(node.id)
                elif access_count == 0:
                    cold_nodes.append(node.id)
            
            # Consistency checks
            orphaned_edges, circular_deps, integrity_score = self._check_consistency(graph)
            
            # Create canonical state
            state = CanonicalStateSchema(
                state_id=f"state_{int(start_time.timestamp())}",
                version=self.version.value,
                node_count=node_count,
                edge_count=edge_count,
                node_type_counts=node_type_counts,
                edge_type_counts=edge_type_counts,
                average_node_strength=avg_node_strength,
                average_edge_strength=avg_edge_strength,
                total_graph_strength=total_strength,
                oldest_node_age_hours=oldest_age,
                newest_node_age_hours=newest_age,
                cluster_count=cluster_count,
                largest_cluster_size=largest_cluster_size,
                access_patterns=access_patterns,
                hot_nodes=hot_nodes[:10],  # Limit to top 10
                cold_nodes=cold_nodes[:10],  # Limit to top 10
                orphaned_edges=orphaned_edges,
                circular_dependencies=circular_deps,
                integrity_score=integrity_score
            )
            
            derivation_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"Derived canonical state in {derivation_time:.2f}ms")
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to derive canonical state: {e}")
            raise PlaygroundException(f"State derivation failed: {e}")
    
    def _analyze_clusters(self, graph: SemanticGraph) -> List[List[str]]:
        """Analyze graph clusters using simple connected components."""
        visited = set()
        clusters = []
        
        def dfs(node_id: str, cluster: List[str]) -> None:
            if node_id in visited:
                return
            
            visited.add(node_id)
            cluster.append(node_id)
            
            # Visit connected nodes
            for edge in graph.edges.values():
                if edge.source_id == node_id and edge.target_id not in visited:
                    dfs(edge.target_id, cluster)
                elif edge.target_id == node_id and edge.source_id not in visited:
                    dfs(edge.source_id, cluster)
        
        for node_id in graph.nodes.keys():
            if node_id not in visited:
                cluster = []
                dfs(node_id, cluster)
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _check_consistency(self, graph: SemanticGraph) -> Tuple[int, int, float]:
        """Check graph consistency and return metrics."""
        orphaned_edges = 0
        circular_deps = 0
        total_checks = 0
        passed_checks = 0
        
        # Check for orphaned edges
        for edge in graph.edges.values():
            total_checks += 1
            if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes:
                orphaned_edges += 1
            else:
                passed_checks += 1
        
        # Check for circular dependencies (simple cycle detection)
        if self.consistency_level in [ConsistencyLevel.STRICT, ConsistencyLevel.PARANOID]:
            circular_deps = self._detect_cycles(graph)
            total_checks += len(graph.nodes)
            passed_checks += len(graph.nodes) - circular_deps
        
        # Additional paranoid checks
        if self.consistency_level == ConsistencyLevel.PARANOID:
            # Check node metadata consistency
            for node in graph.nodes.values():
                total_checks += 1
                if (node.metadata.strength < 0 or node.metadata.strength > 2.0 or
                    node.metadata.access_count < 0):
                    # Inconsistent metadata
                    pass
                else:
                    passed_checks += 1
        
        integrity_score = passed_checks / total_checks if total_checks > 0 else 1.0
        return orphaned_edges, circular_deps, integrity_score
    
    def _detect_cycles(self, graph: SemanticGraph) -> int:
        """Simple cycle detection using DFS."""
        visited = set()
        rec_stack = set()
        cycles = 0
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Check all adjacent nodes
            for edge in graph.edges.values():
                if edge.source_id == node_id:
                    neighbor = edge.target_id
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in graph.nodes.keys():
            if node_id not in visited:
                if has_cycle(node_id):
                    cycles += 1
        
        return cycles


class StructuredStateLayer:
    """
    Layer 2 of LPRA - Manages canonical state derived from semantic graph.
    Provides deterministic state computation, storage, and version management.
    """
    
    def __init__(self, 
                 sqlite_path: Path,
                 lancedb_path: Path,
                 consistency_level: ConsistencyLevel = ConsistencyLevel.STRICT):
        """Initialize the structured state layer."""
        self.sqlite_store = SQLiteStateStore(sqlite_path)
        self.lancedb_store = LanceDBStateStore(lancedb_path)
        self.derivation_engine = DerivationEngine(consistency_level)
        
        logger.info("StructuredStateLayer initialized")
    
    def compute_and_store_state(self, graph: SemanticGraph, state_id: Optional[str] = None) -> str:
        """Compute canonical state from graph and store it."""
        start_time = datetime.now()
        
        # Derive canonical state
        canonical_state = self.derivation_engine.derive_state(graph)
        
        if state_id is None:
            state_id = canonical_state.state_id
        
        # Compute metadata
        state_dict = canonical_state.dict()
        state_json = json.dumps(state_dict, sort_keys=True)
        checksum = hashlib.sha256(state_json.encode()).hexdigest()
        
        graph_data = graph.export_to_dict()
        graph_json = json.dumps(graph_data, sort_keys=True)
        graph_hash = hashlib.sha256(graph_json.encode()).hexdigest()
        
        derivation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        metadata = StateMetadata(
            version=self.derivation_engine.version,
            created_at=start_time,
            updated_at=datetime.now(),
            checksum=checksum,
            consistency_level=self.derivation_engine.consistency_level,
            source_graph_hash=graph_hash,
            derivation_time_ms=derivation_time
        )
        
        # Store in both backends
        sqlite_success = self.sqlite_store.save_state(state_id, state_dict, metadata)
        lancedb_success = self.lancedb_store.save_state(state_id, state_dict, metadata)
        
        if sqlite_success and lancedb_success:
            logger.info(f"Successfully stored state {state_id}")
            return state_id
        else:
            logger.error(f"Failed to store state {state_id}")
            raise PlaygroundException(f"Failed to store state {state_id}")
    
    def load_state(self, state_id: str) -> Optional[CanonicalStateSchema]:
        """Load a canonical state by ID."""
        # Try SQLite first (faster for exact lookups)
        result = self.sqlite_store.load_state(state_id)
        
        if result:
            state_data, metadata = result
            try:
                return CanonicalStateSchema(**state_data)
            except Exception as e:
                logger.error(f"Failed to validate loaded state {state_id}: {e}")
                return None
        
        logger.warning(f"State {state_id} not found")
        return None
    
    def find_similar_states(self, target_state: CanonicalStateSchema, limit: int = 5) -> List[Tuple[str, float]]:
        """Find states similar to the target state using vector similarity."""
        # This would use LanceDB's vector search capabilities
        # For now, return empty list as placeholder
        logger.info(f"Searching for states similar to {target_state.state_id}")
        return []
    
    def list_recent_states(self, limit: int = 10) -> List[Tuple[str, StateMetadata]]:
        """List recent state snapshots."""
        return self.sqlite_store.list_states(limit)
    
    def cleanup_old_states(self, keep_count: int = 50) -> int:
        """Clean up old state snapshots from both stores."""
        sqlite_cleaned = self.sqlite_store.cleanup_old_states(keep_count)
        lancedb_cleaned = self.lancedb_store.cleanup_old_states(keep_count)
        
        total_cleaned = sqlite_cleaned + lancedb_cleaned
        logger.info(f"Cleaned up {total_cleaned} old state snapshots")
        return total_cleaned
    
    def validate_state_integrity(self, state_id: str) -> bool:
        """Validate the integrity of a stored state."""
        result = self.sqlite_store.load_state(state_id)
        if not result:
            return False
        
        state_data, metadata = result
        
        # Recompute checksum
        state_json = json.dumps(state_data, sort_keys=True)
        computed_checksum = hashlib.sha256(state_json.encode()).hexdigest()
        
        return computed_checksum == metadata.checksum
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored states."""
        states = self.list_recent_states(limit=100)
        
        if not states:
            return {"total_states": 0}
        
        versions = {}
        consistency_levels = {}
        avg_derivation_time = 0
        
        for state_id, metadata in states:
            versions[metadata.version.value] = versions.get(metadata.version.value, 0) + 1
            consistency_levels[metadata.consistency_level.value] = consistency_levels.get(metadata.consistency_level.value, 0) + 1
            avg_derivation_time += metadata.derivation_time_ms
        
        avg_derivation_time /= len(states)
        
        return {
            "total_states": len(states),
            "version_distribution": versions,
            "consistency_level_distribution": consistency_levels,
            "average_derivation_time_ms": avg_derivation_time,
            "oldest_state": min(states, key=lambda x: x[1].created_at)[1].created_at.isoformat(),
            "newest_state": max(states, key=lambda x: x[1].created_at)[1].created_at.isoformat()
        }