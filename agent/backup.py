"""
Backup and Recovery System for Proof-of-Work Agent.
Provides automated backups, corruption detection, and disaster recovery.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
from threading import Lock

from agent.logger import get_logger


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"           # Complete backup
    INCREMENTAL = "incremental"  # Only changed files
    SNAPSHOT = "snapshot"   # Point-in-time state


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupManifest:
    """Backup manifest with metadata."""
    
    id: str
    backup_type: str
    created_at: str
    source_path: str
    backup_path: str
    files_count: int
    total_size: int
    compressed_size: int
    checksum: str
    status: str
    files: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupManifest":
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save manifest to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "BackupManifest":
        """Load manifest from file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class RecoveryPoint:
    """Recovery point for rollback."""
    
    id: str
    created_at: str
    description: str
    backup_path: str
    state_hash: str
    can_rollback: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackupManager:
    """
    Manages backups for the POW Agent.
    
    Features:
    - Full and incremental backups
    - Compression with gzip
    - Integrity verification
    - Automatic cleanup of old backups
    - Recovery point management
    """
    
    MODULE = "backup"
    MANIFEST_FILE = "manifest.json"
    
    def __init__(
        self,
        source_dir: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        max_backups: int = 10,
        max_age_days: int = 7
    ):
        self.log = get_logger(self.MODULE)
        
        self.source_dir = source_dir or Path(__file__).resolve().parents[1]
        self.backup_dir = backup_dir or self.source_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_backups = max_backups
        self.max_age_days = max_age_days
        
        self._lock = Lock()
        self._backup_in_progress = False
        
        # Patterns for files to backup
        self.include_patterns = {
            "data/*.json",
            "logs/*.log",
            "state.json",
            ".env",
            "tasks/*.json",
        }
        
        # Patterns to exclude
        self.exclude_patterns = {
            "__pycache__",
            "*.pyc",
            ".git",
            "node_modules",
            "backups",
            "*.tmp",
        }
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}"
    
    def _get_file_hash(self, path: Path) -> str:
        """Calculate file hash."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _should_include(self, path: Path) -> bool:
        """Check if file should be included in backup."""
        path_str = str(path)
        
        # Check exclusions
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return False
        
        return True
    
    def _collect_files(self) -> List[Dict[str, Any]]:
        """Collect files to backup."""
        files = []
        
        for path in self.source_dir.rglob("*"):
            if not path.is_file():
                continue
            
            if not self._should_include(path):
                continue
            
            rel_path = path.relative_to(self.source_dir)
            
            try:
                stat = path.stat()
                file_hash = self._get_file_hash(path)
                
                files.append({
                    "path": str(rel_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "hash": file_hash
                })
            except Exception as e:
                self.log.warn(f"Failed to process {path}: {e}")
        
        return files
    
    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        description: str = ""
    ) -> Optional[BackupManifest]:
        """
        Create a new backup.
        
        Args:
            backup_type: Type of backup to create
            description: Optional description
            
        Returns:
            BackupManifest if successful, None otherwise
        """
        with self._lock:
            if self._backup_in_progress:
                self.log.warn("Backup already in progress")
                return None
            self._backup_in_progress = True
        
        try:
            backup_id = self._generate_backup_id()
            backup_path = self.backup_dir / f"{backup_id}.tar.gz"
            manifest_path = self.backup_dir / f"{backup_id}.manifest.json"
            
            self.log.info(f"Creating {backup_type.value} backup: {backup_id}")
            
            # Collect files
            files = self._collect_files()
            total_size = sum(f["size"] for f in files)
            
            # Create tarball
            with tarfile.open(backup_path, "w:gz") as tar:
                for file_info in files:
                    file_path = self.source_dir / file_info["path"]
                    tar.add(file_path, arcname=file_info["path"])
            
            # Calculate compressed size and checksum
            compressed_size = backup_path.stat().st_size
            backup_hash = self._get_file_hash(backup_path)
            
            # Create manifest
            manifest = BackupManifest(
                id=backup_id,
                backup_type=backup_type.value,
                created_at=datetime.now(timezone.utc).isoformat(),
                source_path=str(self.source_dir),
                backup_path=str(backup_path),
                files_count=len(files),
                total_size=total_size,
                compressed_size=compressed_size,
                checksum=backup_hash,
                status=BackupStatus.COMPLETED.value,
                files=files,
                metadata={"description": description}
            )
            
            manifest.save(manifest_path)
            
            self.log.info(
                f"Backup completed: {len(files)} files, "
                f"{total_size / 1024:.1f}KB -> {compressed_size / 1024:.1f}KB"
            )
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return manifest
            
        except Exception as e:
            self.log.error(f"Backup failed: {e}")
            return None
        
        finally:
            with self._lock:
                self._backup_in_progress = False
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        manifest_path = self.backup_dir / f"{backup_id}.manifest.json"
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        if not manifest_path.exists() or not backup_path.exists():
            self.log.error(f"Backup {backup_id} not found")
            return False
        
        try:
            manifest = BackupManifest.load(manifest_path)
            
            # Verify checksum
            actual_hash = self._get_file_hash(backup_path)
            if actual_hash != manifest.checksum:
                self.log.error(f"Checksum mismatch for {backup_id}")
                return False
            
            # Verify archive integrity
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.getmembers()
            
            self.log.info(f"Backup {backup_id} verified successfully")
            return True
            
        except Exception as e:
            self.log.error(f"Backup verification failed: {e}")
            return False
    
    def restore_backup(
        self,
        backup_id: str,
        target_dir: Optional[Path] = None,
        files: Optional[List[str]] = None
    ) -> bool:
        """
        Restore from backup.
        
        Args:
            backup_id: ID of backup to restore
            target_dir: Target directory (default: source_dir)
            files: Specific files to restore (default: all)
            
        Returns:
            True if successful
        """
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        if not backup_path.exists():
            self.log.error(f"Backup {backup_id} not found")
            return False
        
        if not self.verify_backup(backup_id):
            self.log.error("Backup verification failed, aborting restore")
            return False
        
        target = target_dir or self.source_dir
        
        try:
            self.log.info(f"Restoring backup {backup_id} to {target}")
            
            with tarfile.open(backup_path, "r:gz") as tar:
                if files:
                    # Restore specific files
                    for member in tar.getmembers():
                        if member.name in files:
                            tar.extract(member, target)
                else:
                    # Restore all
                    tar.extractall(target)
            
            self.log.info("Restore completed successfully")
            return True
            
        except Exception as e:
            self.log.error(f"Restore failed: {e}")
            return False
    
    def list_backups(self) -> List[BackupManifest]:
        """List all available backups."""
        backups = []
        
        for manifest_path in self.backup_dir.glob("*.manifest.json"):
            try:
                manifest = BackupManifest.load(manifest_path)
                backups.append(manifest)
            except Exception as e:
                self.log.warn(f"Failed to load manifest {manifest_path}: {e}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        manifest_path = self.backup_dir / f"{backup_id}.manifest.json"
        
        try:
            if backup_path.exists():
                backup_path.unlink()
            if manifest_path.exists():
                manifest_path.unlink()
            
            self.log.info(f"Deleted backup {backup_id}")
            return True
            
        except Exception as e:
            self.log.error(f"Failed to delete backup: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy."""
        backups = self.list_backups()
        
        # Remove by count
        if len(backups) > self.max_backups:
            for backup in backups[self.max_backups:]:
                self.delete_backup(backup.id)
        
        # Remove by age
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        for backup in backups:
            created = datetime.fromisoformat(backup.created_at.replace('Z', '+00:00'))
            if created < cutoff:
                self.delete_backup(backup.id)
    
    def get_latest_backup(self) -> Optional[BackupManifest]:
        """Get the most recent backup."""
        backups = self.list_backups()
        return backups[0] if backups else None


class RecoveryManager:
    """
    Manages recovery points and disaster recovery.
    """
    
    MODULE = "recovery"
    
    def __init__(self, backup_manager: BackupManager):
        self.log = get_logger(self.MODULE)
        self.backup_manager = backup_manager
        self.recovery_points: List[RecoveryPoint] = []
        self._load_recovery_points()
    
    def _recovery_points_path(self) -> Path:
        """Get recovery points file path."""
        return self.backup_manager.backup_dir / "recovery_points.json"
    
    def _load_recovery_points(self) -> None:
        """Load recovery points from file."""
        path = self._recovery_points_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.recovery_points = [RecoveryPoint(**rp) for rp in data]
            except Exception as e:
                self.log.error(f"Failed to load recovery points: {e}")
    
    def _save_recovery_points(self) -> None:
        """Save recovery points to file."""
        path = self._recovery_points_path()
        with open(path, 'w') as f:
            json.dump([rp.to_dict() for rp in self.recovery_points], f, indent=2)
    
    def create_recovery_point(self, description: str = "Manual recovery point") -> Optional[RecoveryPoint]:
        """Create a new recovery point."""
        backup = self.backup_manager.create_backup(
            backup_type=BackupType.SNAPSHOT,
            description=description
        )
        
        if not backup:
            return None
        
        recovery_point = RecoveryPoint(
            id=backup.id,
            created_at=backup.created_at,
            description=description,
            backup_path=backup.backup_path,
            state_hash=backup.checksum,
            can_rollback=True
        )
        
        self.recovery_points.append(recovery_point)
        self._save_recovery_points()
        
        self.log.info(f"Created recovery point: {recovery_point.id}")
        return recovery_point
    
    def rollback_to(self, recovery_point_id: str) -> bool:
        """Rollback to a specific recovery point."""
        point = None
        for rp in self.recovery_points:
            if rp.id == recovery_point_id:
                point = rp
                break
        
        if not point:
            self.log.error(f"Recovery point {recovery_point_id} not found")
            return False
        
        if not point.can_rollback:
            self.log.error(f"Recovery point {recovery_point_id} cannot be rolled back")
            return False
        
        # Create a new recovery point before rollback
        self.create_recovery_point(f"Pre-rollback to {recovery_point_id}")
        
        # Perform restore
        return self.backup_manager.restore_backup(recovery_point_id)
    
    def list_recovery_points(self) -> List[RecoveryPoint]:
        """List all recovery points."""
        return sorted(self.recovery_points, key=lambda rp: rp.created_at, reverse=True)


class StateSnapshot:
    """
    Quick state snapshots for rapid recovery.
    Lighter weight than full backups.
    """
    
    MODULE = "snapshot"
    SNAPSHOTS_DIR = "snapshots"
    MAX_SNAPSHOTS = 20
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.log = get_logger(self.MODULE)
        self.data_dir = data_dir or Path(__file__).resolve().parents[1] / "data"
        self.snapshots_dir = self.data_dir / self.SNAPSHOTS_DIR
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    def capture(self, name: str = "") -> str:
        """Capture current state snapshot."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        name = name or f"snapshot_{timestamp}"
        snapshot_path = self.snapshots_dir / f"{name}.json.gz"
        
        # Collect state data
        state_data = {
            "timestamp": timestamp,
            "name": name,
            "files": {}
        }
        
        # Capture JSON files from data directory
        for json_file in self.data_dir.glob("*.json"):
            if json_file.parent == self.snapshots_dir:
                continue
            
            try:
                with open(json_file, 'r') as f:
                    state_data["files"][json_file.name] = json.load(f)
            except Exception as e:
                self.log.warn(f"Failed to capture {json_file}: {e}")
        
        # Save compressed
        with gzip.open(snapshot_path, 'wt', encoding='utf-8') as f:
            json.dump(state_data, f)
        
        self.log.debug(f"Captured snapshot: {name}")
        self._cleanup()
        
        return name
    
    def restore(self, name: str) -> bool:
        """Restore from snapshot."""
        snapshot_path = self.snapshots_dir / f"{name}.json.gz"
        
        if not snapshot_path.exists():
            self.log.error(f"Snapshot {name} not found")
            return False
        
        try:
            with gzip.open(snapshot_path, 'rt', encoding='utf-8') as f:
                state_data = json.load(f)
            
            for filename, content in state_data["files"].items():
                file_path = self.data_dir / filename
                with open(file_path, 'w') as f:
                    json.dump(content, f, indent=2)
            
            self.log.info(f"Restored snapshot: {name}")
            return True
            
        except Exception as e:
            self.log.error(f"Failed to restore snapshot: {e}")
            return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        snapshots = []
        
        for path in sorted(self.snapshots_dir.glob("*.json.gz"), reverse=True):
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                snapshots.append({
                    "name": data.get("name", path.stem),
                    "timestamp": data.get("timestamp", ""),
                    "size": path.stat().st_size,
                    "files": len(data.get("files", {}))
                })
            except Exception:
                continue
        
        return snapshots
    
    def _cleanup(self) -> None:
        """Remove old snapshots."""
        snapshots = sorted(
            self.snapshots_dir.glob("*.json.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_snapshot in snapshots[self.MAX_SNAPSHOTS:]:
            old_snapshot.unlink()


# ============================================================
# Convenience Functions
# ============================================================

_backup_manager: Optional[BackupManager] = None
_state_snapshot: Optional[StateSnapshot] = None


def get_backup_manager() -> BackupManager:
    """Get global backup manager."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


def get_state_snapshot() -> StateSnapshot:
    """Get global state snapshot manager."""
    global _state_snapshot
    if _state_snapshot is None:
        _state_snapshot = StateSnapshot()
    return _state_snapshot


def quick_backup() -> Optional[BackupManifest]:
    """Create a quick backup."""
    return get_backup_manager().create_backup()


def quick_snapshot(name: str = "") -> str:
    """Create a quick state snapshot."""
    return get_state_snapshot().capture(name)
