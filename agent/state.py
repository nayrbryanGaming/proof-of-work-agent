"""
State Manager for persistent agent state.
Handles state persistence, recovery, and metrics tracking.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock

from agent.logger import get_logger


@dataclass
class CycleMetrics:
    """Metrics for a single cycle."""
    cycle_number: int
    started_at: str
    completed_at: Optional[str] = None
    duration: float = 0.0
    heartbeat_synced: bool = False
    status_checked: bool = False
    forum_engaged: bool = False
    posts_voted: int = 0
    posts_commented: int = 0
    task_solved: bool = False
    task_hash: Optional[str] = None
    solana_tx: Optional[str] = None
    project_updated: bool = False
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentMetrics:
    """Aggregate agent metrics."""
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    tasks_solved: int = 0
    forum_engagements: int = 0
    total_votes: int = 0
    total_comments: int = 0
    solana_transactions: int = 0
    total_errors: int = 0
    cycle_durations: List[float] = field(default_factory=list)
    
    @property
    def average_cycle_duration(self) -> float:
        if not self.cycle_durations:
            return 0.0
        return sum(self.cycle_durations) / len(self.cycle_durations)
    
    @property
    def success_rate(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return (self.successful_cycles / self.total_cycles) * 100
    
    @property
    def error_rate(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return (self.failed_cycles / self.total_cycles) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['average_cycle_duration'] = self.average_cycle_duration
        d['success_rate'] = self.success_rate
        d['error_rate'] = self.error_rate
        return d


@dataclass
class AgentState:
    """Complete agent state."""
    version: str = "1.0.0"
    agent_id: str = ""
    started_at: Optional[str] = None
    last_active_at: Optional[str] = None
    running: bool = False
    current_cycle: int = 0
    last_heartbeat_at: Optional[str] = None
    last_heartbeat_hash: Optional[str] = None
    last_forum_at: Optional[str] = None
    last_solana_tx: Optional[str] = None
    last_project_update_at: Optional[str] = None
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    recent_cycles: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'agent_id': self.agent_id,
            'started_at': self.started_at,
            'last_active_at': self.last_active_at,
            'running': self.running,
            'current_cycle': self.current_cycle,
            'last_heartbeat_at': self.last_heartbeat_at,
            'last_heartbeat_hash': self.last_heartbeat_hash,
            'last_forum_at': self.last_forum_at,
            'last_solana_tx': self.last_solana_tx,
            'last_project_update_at': self.last_project_update_at,
            'metrics': self.metrics.to_dict(),
            'recent_cycles': self.recent_cycles,
            'error_log': self.error_log[-100:],  # Keep last 100 errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        metrics_data = data.pop('metrics', {})
        recent_cycles = data.pop('recent_cycles', [])
        error_log = data.pop('error_log', [])
        
        metrics = AgentMetrics(
            total_cycles=metrics_data.get('total_cycles', 0),
            successful_cycles=metrics_data.get('successful_cycles', 0),
            failed_cycles=metrics_data.get('failed_cycles', 0),
            tasks_solved=metrics_data.get('tasks_solved', 0),
            forum_engagements=metrics_data.get('forum_engagements', 0),
            total_votes=metrics_data.get('total_votes', 0),
            total_comments=metrics_data.get('total_comments', 0),
            solana_transactions=metrics_data.get('solana_transactions', 0),
            total_errors=metrics_data.get('total_errors', 0),
            cycle_durations=metrics_data.get('cycle_durations', [])[-100:],
        )
        
        return cls(
            metrics=metrics,
            recent_cycles=recent_cycles[-50:],
            error_log=error_log[-100:],
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )


class StateManager:
    """
    Manages persistent agent state with atomic operations.
    Thread-safe and supports recovery from crashes.
    """
    
    MODULE = "state"
    MAX_RECENT_CYCLES = 50
    MAX_ERROR_LOG = 100
    
    def __init__(self, state_file: Optional[Path] = None):
        self.log = get_logger(self.MODULE)
        self.state_file = state_file or Path(__file__).resolve().parent.parent / "data" / "state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._state: AgentState = AgentState()
        self._lock = Lock()
        self._dirty = False
        self._save_task: Optional[asyncio.Task] = None
        
        # Generate unique agent ID
        self._state.agent_id = self._generate_agent_id()
        
        # Load existing state
        self._load()
    
    def _generate_agent_id(self) -> str:
        """Generate a unique agent identifier."""
        import socket
        import os
        
        data = f"{socket.gethostname()}-{os.getpid()}-{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _load(self):
        """Load state from file."""
        if not self.state_file.exists():
            self.log.info("No existing state file, starting fresh")
            return
        
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._state = AgentState.from_dict(data)
            self._state.running = False  # Always start as not running
            self.log.info(f"Loaded state: cycle={self._state.current_cycle}, tasks_solved={self._state.metrics.tasks_solved}")
            
        except Exception as e:
            self.log.error(f"Failed to load state: {e}")
            # Backup corrupted file
            if self.state_file.exists():
                backup = self.state_file.with_suffix('.json.bak')
                self.state_file.rename(backup)
                self.log.warn(f"Backed up corrupted state to {backup}")
    
    def _save(self):
        """Save state to file atomically."""
        try:
            # Write to temp file first
            temp_file = self.state_file.with_suffix('.json.tmp')
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, indent=2, default=str)
            
            # Atomic rename
            temp_file.replace(self.state_file)
            self._dirty = False
            
        except Exception as e:
            self.log.error(f"Failed to save state: {e}")
    
    def save(self):
        """Public save method with lock."""
        with self._lock:
            self._save()
    
    async def save_async(self):
        """Async save that debounces writes."""
        self._dirty = True
        
        # Cancel existing save task
        if self._save_task and not self._save_task.done():
            return  # Already scheduled
        
        async def delayed_save():
            await asyncio.sleep(1.0)  # Debounce
            if self._dirty:
                with self._lock:
                    self._save()
        
        self._save_task = asyncio.create_task(delayed_save())
    
    @property
    def state(self) -> AgentState:
        """Get current state (read-only view)."""
        return self._state
    
    def start_agent(self):
        """Mark agent as started."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            self._state.started_at = now
            self._state.last_active_at = now
            self._state.running = True
            self._save()
        
        self.log.info("Agent started")
    
    def stop_agent(self):
        """Mark agent as stopped."""
        with self._lock:
            self._state.running = False
            self._state.last_active_at = datetime.now(timezone.utc).isoformat()
            self._save()
        
        self.log.info("Agent stopped")
    
    def start_cycle(self) -> CycleMetrics:
        """Start a new cycle and return metrics tracker."""
        with self._lock:
            self._state.current_cycle += 1
            self._state.last_active_at = datetime.now(timezone.utc).isoformat()
            
            metrics = CycleMetrics(
                cycle_number=self._state.current_cycle,
                started_at=datetime.now(timezone.utc).isoformat()
            )
            
            return metrics
    
    def complete_cycle(self, metrics: CycleMetrics):
        """Complete a cycle and update state."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            metrics.completed_at = now
            
            self._state.last_active_at = now
            self._state.metrics.total_cycles += 1
            
            if metrics.heartbeat_synced:
                self._state.last_heartbeat_at = now
            
            if metrics.forum_engaged:
                self._state.last_forum_at = now
                self._state.metrics.forum_engagements += 1
                self._state.metrics.total_votes += metrics.posts_voted
                self._state.metrics.total_comments += metrics.posts_commented
            
            if metrics.task_solved:
                self._state.metrics.tasks_solved += 1
            
            if metrics.solana_tx:
                self._state.last_solana_tx = metrics.solana_tx
                self._state.metrics.solana_transactions += 1
            
            if metrics.project_updated:
                self._state.last_project_update_at = now
            
            if metrics.errors:
                self._state.metrics.failed_cycles += 1
                self._state.metrics.total_errors += len(metrics.errors)
                
                for error in metrics.errors:
                    self._state.error_log.append({
                        'cycle': metrics.cycle_number,
                        'timestamp': now,
                        'error': error
                    })
                
                # Trim error log
                self._state.error_log = self._state.error_log[-self.MAX_ERROR_LOG:]
            else:
                self._state.metrics.successful_cycles += 1
            
            self._state.metrics.cycle_durations.append(metrics.duration)
            self._state.metrics.cycle_durations = self._state.metrics.cycle_durations[-100:]
            
            # Add to recent cycles
            self._state.recent_cycles.append(metrics.to_dict())
            self._state.recent_cycles = self._state.recent_cycles[-self.MAX_RECENT_CYCLES:]
            
            self._save()
        
        self.log.info(
            f"Cycle {metrics.cycle_number} complete: "
            f"duration={metrics.duration:.2f}s, errors={len(metrics.errors)}"
        )
    
    def update_heartbeat(self, content_hash: str):
        """Update heartbeat tracking."""
        with self._lock:
            self._state.last_heartbeat_at = datetime.now(timezone.utc).isoformat()
            self._state.last_heartbeat_hash = content_hash
    
    def record_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """Record an error."""
        with self._lock:
            self._state.error_log.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': error,
                'context': context or {}
            })
            self._state.error_log = self._state.error_log[-self.MAX_ERROR_LOG:]
            self._state.metrics.total_errors += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get state summary for API."""
        with self._lock:
            return {
                'running': self._state.running,
                'cycle_count': self._state.current_cycle,
                'last_cycle_at': self._state.recent_cycles[-1]['completed_at'] if self._state.recent_cycles else None,
                'last_heartbeat_at': self._state.last_heartbeat_at,
                'last_forum_at': self._state.last_forum_at,
                'last_solana_tx': self._state.last_solana_tx,
                'errors_count': self._state.metrics.total_errors,
                'tasks_solved': self._state.metrics.tasks_solved,
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for API."""
        with self._lock:
            started = self._state.started_at
            if started:
                start_dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
                uptime = (datetime.now(timezone.utc) - start_dt).total_seconds()
            else:
                uptime = 0
            
            return {
                'uptime_seconds': uptime,
                **self._state.metrics.to_dict()
            }


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get or create the global state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
