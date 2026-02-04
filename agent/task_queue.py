"""
Task Queue System - Persistent task queue with priorities and scheduling.
Supports delayed execution, retries, and dead letter queue.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from threading import Lock
import hashlib

from agent.logger import get_logger


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead_letter"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class TaskDefinition:
    """Task definition with all metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    
    # Timing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scheduled_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 5.0
    
    # Execution
    timeout: float = 300.0  # 5 minutes default
    result: Any = None
    error: Optional[str] = None
    
    # Tracking
    idempotency_key: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        if not isinstance(other, TaskDefinition):
            return NotImplemented
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "scheduled_at": self.scheduled_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "result": self.result,
            "error": self.error,
            "idempotency_key": self.idempotency_key,
            "correlation_id": self.correlation_id,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDefinition":
        """Create from dictionary."""
        data = dict(data)
        if "priority" in data:
            data["priority"] = TaskPriority(data["priority"])
        if "status" in data:
            data["status"] = TaskStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TaskHandler(ABC):
    """Abstract base class for task handlers."""
    
    @abstractmethod
    async def handle(self, task: TaskDefinition) -> Any:
        """Handle a task and return result."""
        pass


class FunctionTaskHandler(TaskHandler):
    """Task handler using a function."""
    
    def __init__(self, func: Callable):
        self.func = func
    
    async def handle(self, task: TaskDefinition) -> Any:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(task)
        return self.func(task)


class TaskQueue:
    """
    Persistent task queue with priorities and scheduling.
    """
    
    def __init__(
        self,
        name: str = "default",
        persistence_path: Optional[Path] = None,
        max_concurrent: int = 5
    ):
        self.name = name
        self.log = get_logger(f"task_queue.{name}")
        self.max_concurrent = max_concurrent
        
        # Storage
        self._queue: List[TaskDefinition] = []  # Priority heap
        self._scheduled: Dict[str, TaskDefinition] = {}  # Scheduled tasks
        self._running: Dict[str, TaskDefinition] = {}  # Currently running
        self._completed: Dict[str, TaskDefinition] = {}  # Completed (limited history)
        self._dead_letter: Dict[str, TaskDefinition] = {}  # Failed tasks
        self._idempotency_cache: Set[str] = set()  # Processed idempotency keys
        
        self._lock = Lock()
        self._handlers: Dict[str, TaskHandler] = {}
        self._default_handler: Optional[TaskHandler] = None
        
        # Persistence
        self.persistence_path = persistence_path or Path(__file__).parent.parent / "data" / "tasks"
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Load persisted tasks
        self._load()
    
    def _load(self):
        """Load tasks from persistence."""
        queue_file = self.persistence_path / f"{self.name}_queue.json"
        if queue_file.exists():
            try:
                with queue_file.open("r") as f:
                    data = json.load(f)
                
                for task_data in data.get("queue", []):
                    task = TaskDefinition.from_dict(task_data)
                    heapq.heappush(self._queue, task)
                
                for task_data in data.get("scheduled", []):
                    task = TaskDefinition.from_dict(task_data)
                    self._scheduled[task.id] = task
                
                self.log.info(
                    f"Loaded {len(self._queue)} queued, "
                    f"{len(self._scheduled)} scheduled tasks"
                )
            except Exception as e:
                self.log.error(f"Failed to load tasks: {e}")
    
    def _save(self):
        """Save tasks to persistence."""
        queue_file = self.persistence_path / f"{self.name}_queue.json"
        try:
            with self._lock:
                data = {
                    "queue": [t.to_dict() for t in self._queue],
                    "scheduled": [t.to_dict() for t in self._scheduled.values()],
                    "dead_letter": [t.to_dict() for t in self._dead_letter.values()]
                }
            
            with queue_file.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log.error(f"Failed to save tasks: {e}")
    
    def register_handler(self, task_name: str, handler: TaskHandler):
        """Register a handler for a task type."""
        with self._lock:
            self._handlers[task_name] = handler
    
    def set_default_handler(self, handler: TaskHandler):
        """Set default handler for unknown task types."""
        self._default_handler = handler
    
    def _check_idempotency(self, task: TaskDefinition) -> bool:
        """Check if task with idempotency key was already processed."""
        if not task.idempotency_key:
            return False
        return task.idempotency_key in self._idempotency_cache
    
    def enqueue(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay: Optional[float] = None,
        schedule_at: Optional[datetime] = None,
        idempotency_key: Optional[str] = None,
        correlation_id: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 300.0,
        tags: Optional[List[str]] = None
    ) -> TaskDefinition:
        """
        Enqueue a new task.
        
        Args:
            name: Task name (used to find handler)
            payload: Task data
            priority: Task priority
            delay: Delay in seconds before executing
            schedule_at: Specific time to execute
            idempotency_key: Key to prevent duplicate processing
            correlation_id: ID for tracking related tasks
            max_retries: Maximum retry attempts
            timeout: Execution timeout in seconds
            tags: Tags for filtering
        
        Returns:
            Created task
        """
        task = TaskDefinition(
            name=name,
            payload=payload or {},
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            tags=tags or []
        )
        
        # Check idempotency
        if self._check_idempotency(task):
            self.log.warn(f"Duplicate task rejected: {idempotency_key}")
            task.status = TaskStatus.CANCELLED
            return task
        
        with self._lock:
            if schedule_at:
                task.status = TaskStatus.SCHEDULED
                task.scheduled_at = schedule_at.isoformat()
                self._scheduled[task.id] = task
            elif delay:
                task.status = TaskStatus.SCHEDULED
                task.scheduled_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
                self._scheduled[task.id] = task
            else:
                heapq.heappush(self._queue, task)
        
        self._save()
        self.log.debug(f"Enqueued task {task.id}: {name}")
        return task
    
    def dequeue(self) -> Optional[TaskDefinition]:
        """Get next task from queue."""
        with self._lock:
            if not self._queue:
                return None
            
            if len(self._running) >= self.max_concurrent:
                return None
            
            task = heapq.heappop(self._queue)
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc).isoformat()
            self._running[task.id] = task
        
        return task
    
    def complete(self, task_id: str, result: Any = None):
        """Mark task as completed."""
        with self._lock:
            if task_id not in self._running:
                return
            
            task = self._running.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.result = result
            
            if task.idempotency_key:
                self._idempotency_cache.add(task.idempotency_key)
            
            # Keep limited history
            self._completed[task_id] = task
            if len(self._completed) > 1000:
                oldest = min(self._completed.values(), key=lambda t: t.completed_at)
                del self._completed[oldest.id]
        
        self._save()
        self.log.debug(f"Completed task {task_id}")
    
    def fail(self, task_id: str, error: str, retry: bool = True):
        """Mark task as failed, optionally retry."""
        with self._lock:
            if task_id not in self._running:
                return
            
            task = self._running.pop(task_id)
            task.error = error
            
            if retry and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.SCHEDULED
                delay = task.retry_delay * (2 ** (task.retry_count - 1))  # Exponential backoff
                task.scheduled_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=delay)
                ).isoformat()
                self._scheduled[task.id] = task
                self.log.warn(
                    f"Task {task_id} failed, retry {task.retry_count}/{task.max_retries} "
                    f"in {delay:.1f}s: {error}"
                )
            else:
                task.status = TaskStatus.DEAD_LETTER
                task.completed_at = datetime.now(timezone.utc).isoformat()
                self._dead_letter[task_id] = task
                self.log.error(f"Task {task_id} moved to dead letter queue: {error}")
        
        self._save()
    
    def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            # Check scheduled
            if task_id in self._scheduled:
                task = self._scheduled.pop(task_id)
                task.status = TaskStatus.CANCELLED
                return True
            
            # Check queue
            for i, task in enumerate(self._queue):
                if task.id == task_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    task.status = TaskStatus.CANCELLED
                    return True
        
        return False
    
    def _process_scheduled(self):
        """Move scheduled tasks that are due to the queue."""
        now = datetime.now(timezone.utc)
        to_move = []
        
        with self._lock:
            for task_id, task in self._scheduled.items():
                if task.scheduled_at:
                    scheduled_time = datetime.fromisoformat(task.scheduled_at.replace('Z', '+00:00'))
                    if scheduled_time <= now:
                        to_move.append(task_id)
            
            for task_id in to_move:
                task = self._scheduled.pop(task_id)
                task.status = TaskStatus.PENDING
                heapq.heappush(self._queue, task)
        
        if to_move:
            self.log.debug(f"Moved {len(to_move)} scheduled tasks to queue")
    
    async def process_one(self) -> Optional[TaskDefinition]:
        """Process a single task."""
        self._process_scheduled()
        
        task = self.dequeue()
        if not task:
            return None
        
        handler = self._handlers.get(task.name) or self._default_handler
        if not handler:
            self.fail(task.id, f"No handler for task type: {task.name}", retry=False)
            return task
        
        try:
            result = await asyncio.wait_for(
                handler.handle(task),
                timeout=task.timeout
            )
            self.complete(task.id, result)
        except asyncio.TimeoutError:
            self.fail(task.id, f"Task timed out after {task.timeout}s")
        except Exception as e:
            self.fail(task.id, str(e))
        
        return task
    
    async def run_worker(self, poll_interval: float = 1.0):
        """Run task worker loop."""
        self.log.info(f"Starting task worker (max_concurrent={self.max_concurrent})")
        
        while True:
            try:
                task = await self.process_one()
                if task:
                    self.log.debug(f"Processed task {task.id}: {task.status.value}")
                else:
                    await asyncio.sleep(poll_interval)
            except Exception as e:
                self.log.error(f"Worker error: {e}")
                await asyncio.sleep(poll_interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "queued": len(self._queue),
                "scheduled": len(self._scheduled),
                "running": len(self._running),
                "completed": len(self._completed),
                "dead_letter": len(self._dead_letter),
                "max_concurrent": self.max_concurrent
            }
    
    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """Get task by ID."""
        with self._lock:
            if task_id in self._running:
                return self._running[task_id]
            if task_id in self._scheduled:
                return self._scheduled[task_id]
            if task_id in self._completed:
                return self._completed[task_id]
            if task_id in self._dead_letter:
                return self._dead_letter[task_id]
            
            for task in self._queue:
                if task.id == task_id:
                    return task
        
        return None
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> List[TaskDefinition]:
        """List tasks, optionally filtered by status."""
        tasks = []
        
        with self._lock:
            if status is None or status == TaskStatus.PENDING:
                tasks.extend(self._queue[:limit])
            if status is None or status == TaskStatus.SCHEDULED:
                tasks.extend(list(self._scheduled.values())[:limit])
            if status is None or status == TaskStatus.RUNNING:
                tasks.extend(list(self._running.values())[:limit])
            if status is None or status == TaskStatus.COMPLETED:
                tasks.extend(list(self._completed.values())[:limit])
            if status is None or status == TaskStatus.DEAD_LETTER:
                tasks.extend(list(self._dead_letter.values())[:limit])
        
        return tasks[:limit]
    
    def retry_dead_letter(self, task_id: str) -> bool:
        """Retry a task from dead letter queue."""
        with self._lock:
            if task_id not in self._dead_letter:
                return False
            
            task = self._dead_letter.pop(task_id)
            task.status = TaskStatus.PENDING
            task.retry_count = 0
            task.error = None
            heapq.heappush(self._queue, task)
        
        self._save()
        self.log.info(f"Retrying dead letter task: {task_id}")
        return True


# Global task queue
task_queue = TaskQueue("agent")


# Convenience decorator
def task(
    name: Optional[str] = None,
    queue: Optional[TaskQueue] = None
):
    """Decorator to register a function as a task handler."""
    def decorator(func: Callable):
        handler = FunctionTaskHandler(func)
        task_name = name or func.__name__
        target_queue = queue or task_queue
        target_queue.register_handler(task_name, handler)
        return func
    return decorator
