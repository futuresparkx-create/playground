"""
LPRA Versioning System
Handles schema versions and migration support for the LPRA architecture.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SchemaVersion:
    """Represents a schema version with metadata"""
    version: str
    description: str
    created_at: datetime
    migration_required: bool = False
    breaking_changes: List[str] = None
    
    def __post_init__(self):
        if self.breaking_changes is None:
            self.breaking_changes = []

class Migration(ABC):
    """Abstract base class for schema migrations"""
    
    @property
    @abstractmethod
    def from_version(self) -> str:
        """Source version for this migration"""
        pass
    
    @property
    @abstractmethod
    def to_version(self) -> str:
        """Target version for this migration"""
        pass
    
    @abstractmethod
    def migrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the migration"""
        pass
    
    @abstractmethod
    def rollback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback the migration if possible"""
        pass

class VersionManager:
    """Manages schema versions and migrations for LPRA components"""
    
    def __init__(self, version_file: Path = None):
        self.version_file = version_file or Path("data/lpra_versions.json")
        self.migrations: Dict[str, Migration] = {}
        self.versions: Dict[str, SchemaVersion] = {}
        self.current_version = "1.0.0"
        
        # Initialize default versions
        self._initialize_default_versions()
        self._load_versions()
    
    def _initialize_default_versions(self):
        """Initialize default schema versions"""
        self.versions = {
            "1.0.0": SchemaVersion(
                version="1.0.0",
                description="Initial LPRA implementation",
                created_at=datetime.now(),
                migration_required=False
            )
        }
    
    def _load_versions(self):
        """Load version information from file"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    self.current_version = data.get("current_version", "1.0.0")
                    
                    # Load version metadata
                    for version_data in data.get("versions", []):
                        version = SchemaVersion(
                            version=version_data["version"],
                            description=version_data["description"],
                            created_at=datetime.fromisoformat(version_data["created_at"]),
                            migration_required=version_data.get("migration_required", False),
                            breaking_changes=version_data.get("breaking_changes", [])
                        )
                        self.versions[version.version] = version
                        
            except Exception as e:
                logger.warning(f"Failed to load version file: {e}")
    
    def _save_versions(self):
        """Save version information to file"""
        try:
            self.version_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "current_version": self.current_version,
                "versions": [
                    {
                        "version": v.version,
                        "description": v.description,
                        "created_at": v.created_at.isoformat(),
                        "migration_required": v.migration_required,
                        "breaking_changes": v.breaking_changes
                    }
                    for v in self.versions.values()
                ]
            }
            
            with open(self.version_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save version file: {e}")
    
    def register_migration(self, migration: Migration):
        """Register a migration"""
        key = f"{migration.from_version}->{migration.to_version}"
        self.migrations[key] = migration
        logger.info(f"Registered migration: {key}")
    
    def add_version(self, version: SchemaVersion):
        """Add a new schema version"""
        self.versions[version.version] = version
        self._save_versions()
        logger.info(f"Added version: {version.version}")
    
    def get_current_version(self) -> str:
        """Get the current schema version"""
        return self.current_version
    
    def set_current_version(self, version: str):
        """Set the current schema version"""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")
        
        self.current_version = version
        self._save_versions()
        logger.info(f"Set current version to: {version}")
    
    def needs_migration(self, data_version: str) -> bool:
        """Check if data needs migration to current version"""
        return data_version != self.current_version
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[Migration]:
        """Get the migration path between two versions"""
        # For now, support direct migrations only
        # In the future, this could implement graph traversal for multi-step migrations
        key = f"{from_version}->{to_version}"
        if key in self.migrations:
            return [self.migrations[key]]
        
        # Try to find a path through intermediate versions
        # This is a simplified implementation - a full version would use graph algorithms
        return []
    
    def migrate_data(self, data: Dict[str, Any], from_version: str, to_version: str = None) -> Dict[str, Any]:
        """Migrate data from one version to another"""
        if to_version is None:
            to_version = self.current_version
        
        if from_version == to_version:
            return data
        
        migration_path = self.get_migration_path(from_version, to_version)
        if not migration_path:
            raise ValueError(f"No migration path from {from_version} to {to_version}")
        
        migrated_data = data.copy()
        for migration in migration_path:
            try:
                migrated_data = migration.migrate(migrated_data)
                logger.info(f"Applied migration: {migration.from_version} -> {migration.to_version}")
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                raise
        
        return migrated_data
    
    def validate_data_version(self, data: Dict[str, Any]) -> str:
        """Extract and validate the version from data"""
        version = data.get("_version", "1.0.0")
        
        if version not in self.versions:
            logger.warning(f"Unknown data version: {version}, assuming 1.0.0")
            return "1.0.0"
        
        return version
    
    def add_version_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add version metadata to data"""
        data_with_version = data.copy()
        data_with_version["_version"] = self.current_version
        data_with_version["_version_timestamp"] = datetime.now().isoformat()
        return data_with_version

# Example migrations for future use
class SemanticGraphV1ToV2Migration(Migration):
    """Example migration for semantic graph schema changes"""
    
    @property
    def from_version(self) -> str:
        return "1.0.0"
    
    @property
    def to_version(self) -> str:
        return "2.0.0"
    
    def migrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate semantic graph from v1 to v2"""
        migrated = data.copy()
        
        # Example: Add new fields, rename existing ones, etc.
        if "nodes" in migrated:
            for node in migrated["nodes"]:
                # Add new required field
                if "created_at" not in node:
                    node["created_at"] = datetime.now().isoformat()
                
                # Rename field
                if "type" in node:
                    node["node_type"] = node.pop("type")
        
        return migrated
    
    def rollback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback v2 to v1"""
        rolled_back = data.copy()
        
        if "nodes" in rolled_back:
            for node in rolled_back["nodes"]:
                # Remove new field
                node.pop("created_at", None)
                
                # Restore old field name
                if "node_type" in node:
                    node["type"] = node.pop("node_type")
        
        return rolled_back

class StructuredStateV1ToV2Migration(Migration):
    """Example migration for structured state schema changes"""
    
    @property
    def from_version(self) -> str:
        return "1.0.0"
    
    @property
    def to_version(self) -> str:
        return "2.0.0"
    
    def migrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate structured state from v1 to v2"""
        migrated = data.copy()
        
        # Example: Restructure state data
        if "state" in migrated:
            state = migrated["state"]
            
            # Group related fields
            if "task_id" in state and "task_description" in state:
                migrated["state"]["task"] = {
                    "id": state.pop("task_id"),
                    "description": state.pop("task_description")
                }
        
        return migrated
    
    def rollback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback v2 to v1"""
        rolled_back = data.copy()
        
        if "state" in rolled_back and "task" in rolled_back["state"]:
            task = rolled_back["state"].pop("task")
            rolled_back["state"]["task_id"] = task.get("id")
            rolled_back["state"]["task_description"] = task.get("description")
        
        return rolled_back

# Global version manager instance
version_manager = VersionManager()

# Register example migrations (for future use)
# version_manager.register_migration(SemanticGraphV1ToV2Migration())
# version_manager.register_migration(StructuredStateV1ToV2Migration())