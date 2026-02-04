"""
Event System - Pub/Sub pattern for decoupled communication.
Enables components to communicate without direct dependencies.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from functools import wraps
from threading import Lock, RLock
from queue import Queue, Empty

from agent.logger import get_logger


class EventPriority(Enum):
    """Priority levels for event handlers."""
    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100
    CRITICAL = 150


class EventType(Enum):
    """Predefined event types."""
    # Lifecycle events
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"
    
    # Cycle events
    CYCLE_STARTED = "cycle.started"
    CYCLE_COMPLETED = "cycle.completed"
    CYCLE_FAILED = "cycle.failed"
    
    # Heartbeat events
    HEARTBEAT_CHECKED = "heartbeat.checked"
    HEARTBEAT_FAILED = "heartbeat.failed"
    
    # Forum events
    FORUM_POST_VOTED = "forum.post.voted"
    FORUM_POST_COMMENTED = "forum.post.commented"
    FORUM_ENGAGEMENT_COMPLETED = "forum.engagement.completed"
    
    # Task events
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    # Solana events
    SOLANA_TX_SUBMITTED = "solana.tx.submitted"
    SOLANA_TX_CONFIRMED = "solana.tx.confirmed"
    SOLANA_TX_FAILED = "solana.tx.failed"
    
    # Project events
    PROJECT_UPDATED = "project.updated"
    
    # State events
    STATE_SAVED = "state.saved"
    STATE_LOADED = "state.loaded"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Event data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: Union[EventType, str] = EventType.CUSTOM
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    propagation_stopped: bool = False
    
    def stop_propagation(self):
        """Stop event from being handled by remaining handlers."""
        self.propagation_stopped = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        event_type = self.type.value if isinstance(self.type, EventType) else self.type
        return {
            "id": self.id,
            "type": event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class EventHandler:
    """Event handler configuration."""
    callback: Callable
    priority: EventPriority = EventPriority.NORMAL
    once: bool = False
    filter_func: Optional[Callable[[Event], bool]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, EventHandler):
            return False
        return self.id == other.id


class EventEmitter:
    """
    Event emitter with pub/sub pattern.
    Supports sync and async handlers, priority ordering, and filters.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.log = get_logger(f"events.{name}")
        
        self._handlers: Dict[str, Set[EventHandler]] = defaultdict(set)
        self._lock = RLock()
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._middleware: List[Callable[[Event], Optional[Event]]] = []
    
    def _get_event_key(self, event_type: Union[EventType, str]) -> str:
        """Get string key for event type."""
        if isinstance(event_type, EventType):
            return event_type.value
        return event_type
    
    def on(
        self,
        event_type: Union[EventType, str],
        callback: Callable,
        priority: EventPriority = EventPriority.NORMAL,
        once: bool = False,
        filter_func: Optional[Callable[[Event], bool]] = None
    ) -> str:
        """
        Register an event handler.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event is emitted
            priority: Handler priority (higher = called first)
            once: If True, handler is removed after first call
            filter_func: Optional filter function
        
        Returns:
            Handler ID for later removal
        """
        handler = EventHandler(
            callback=callback,
            priority=priority,
            once=once,
            filter_func=filter_func
        )
        
        key = self._get_event_key(event_type)
        with self._lock:
            self._handlers[key].add(handler)
        
        self.log.debug(f"Registered handler for {key}: {callback.__name__}")
        return handler.id
    
    def once(
        self,
        event_type: Union[EventType, str],
        callback: Callable,
        priority: EventPriority = EventPriority.NORMAL
    ) -> str:
        """Register a one-time event handler."""
        return self.on(event_type, callback, priority, once=True)
    
    def off(self, event_type: Union[EventType, str], handler_id: str) -> bool:
        """Remove an event handler."""
        key = self._get_event_key(event_type)
        with self._lock:
            handlers = self._handlers.get(key, set())
            for handler in handlers:
                if handler.id == handler_id:
                    handlers.remove(handler)
                    self.log.debug(f"Removed handler {handler_id} for {key}")
                    return True
        return False
    
    def off_all(self, event_type: Optional[Union[EventType, str]] = None):
        """Remove all handlers for an event type, or all handlers if no type specified."""
        with self._lock:
            if event_type is None:
                self._handlers.clear()
                self.log.debug("Removed all handlers")
            else:
                key = self._get_event_key(event_type)
                if key in self._handlers:
                    del self._handlers[key]
                    self.log.debug(f"Removed all handlers for {key}")
    
    def use(self, middleware: Callable[[Event], Optional[Event]]):
        """Add middleware to process events before handlers."""
        with self._lock:
            self._middleware.append(middleware)
    
    def _run_middleware(self, event: Event) -> Optional[Event]:
        """Run middleware chain on event."""
        current = event
        for mw in self._middleware:
            try:
                result = mw(current)
                if result is None:
                    return None  # Middleware cancelled event
                current = result
            except Exception as e:
                self.log.error(f"Middleware error: {e}")
        return current
    
    def _get_sorted_handlers(self, key: str) -> List[EventHandler]:
        """Get handlers sorted by priority (highest first)."""
        with self._lock:
            handlers = list(self._handlers.get(key, set()))
        return sorted(handlers, key=lambda h: h.priority.value, reverse=True)
    
    async def emit_async(
        self,
        event_type: Union[EventType, str],
        data: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Emit an event asynchronously."""
        event = Event(
            type=event_type,
            data=data or {},
            source=source,
            metadata=metadata or {}
        )
        
        # Run middleware
        event = self._run_middleware(event)
        if event is None:
            self.log.debug(f"Event cancelled by middleware")
            return Event(type=event_type)
        
        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
        
        key = self._get_event_key(event_type)
        handlers = self._get_sorted_handlers(key)
        
        # Also get wildcard handlers
        wildcard_handlers = self._get_sorted_handlers("*")
        all_handlers = sorted(
            handlers + wildcard_handlers,
            key=lambda h: h.priority.value,
            reverse=True
        )
        
        handlers_to_remove = []
        
        for handler in all_handlers:
            if event.propagation_stopped:
                break
            
            # Apply filter if present
            if handler.filter_func:
                try:
                    if not handler.filter_func(event):
                        continue
                except Exception as e:
                    self.log.error(f"Filter error: {e}")
                    continue
            
            try:
                if asyncio.iscoroutinefunction(handler.callback):
                    await handler.callback(event)
                else:
                    handler.callback(event)
                
                if handler.once:
                    handlers_to_remove.append((key, handler.id))
                    
            except Exception as e:
                self.log.error(f"Handler error for {key}: {e}")
        
        # Remove one-time handlers
        for event_key, handler_id in handlers_to_remove:
            self.off(event_key, handler_id)
        
        self.log.debug(f"Emitted {key}: handled by {len(all_handlers)} handlers")
        return event
    
    def emit(
        self,
        event_type: Union[EventType, str],
        data: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Emit an event synchronously."""
        event = Event(
            type=event_type,
            data=data or {},
            source=source,
            metadata=metadata or {}
        )
        
        # Run middleware
        event = self._run_middleware(event)
        if event is None:
            return Event(type=event_type)
        
        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
        
        key = self._get_event_key(event_type)
        handlers = self._get_sorted_handlers(key)
        wildcard_handlers = self._get_sorted_handlers("*")
        all_handlers = sorted(
            handlers + wildcard_handlers,
            key=lambda h: h.priority.value,
            reverse=True
        )
        
        handlers_to_remove = []
        
        for handler in all_handlers:
            if event.propagation_stopped:
                break
            
            if handler.filter_func:
                try:
                    if not handler.filter_func(event):
                        continue
                except Exception:
                    continue
            
            try:
                if asyncio.iscoroutinefunction(handler.callback):
                    # Skip async handlers in sync emit
                    self.log.warn(f"Async handler skipped in sync emit: {handler.callback.__name__}")
                    continue
                handler.callback(event)
                
                if handler.once:
                    handlers_to_remove.append((key, handler.id))
                    
            except Exception as e:
                self.log.error(f"Handler error for {key}: {e}")
        
        for event_key, handler_id in handlers_to_remove:
            self.off(event_key, handler_id)
        
        return event
    
    def get_history(
        self,
        event_type: Optional[Union[EventType, str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get event history, optionally filtered by type."""
        with self._lock:
            events = self._event_history[-limit:]
        
        if event_type is not None:
            key = self._get_event_key(event_type)
            events = [e for e in events if self._get_event_key(e.type) == key]
        
        return [e.to_dict() for e in events]
    
    def handler_count(self, event_type: Optional[Union[EventType, str]] = None) -> int:
        """Get count of handlers."""
        with self._lock:
            if event_type is None:
                return sum(len(h) for h in self._handlers.values())
            key = self._get_event_key(event_type)
            return len(self._handlers.get(key, set()))


def on_event(
    event_type: Union[EventType, str],
    emitter: Optional[EventEmitter] = None,
    priority: EventPriority = EventPriority.NORMAL
):
    """Decorator to register a function as an event handler."""
    def decorator(func: Callable):
        target_emitter = emitter or event_bus
        target_emitter.on(event_type, func, priority)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._event_handler_id = func.__name__
        return wrapper
    
    return decorator


class EventQueue:
    """
    Event queue for reliable event processing.
    Events are stored until acknowledged.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.log = get_logger(f"event_queue.{name}")
        
        self._queue: Queue = Queue()
        self._pending: Dict[str, Event] = {}
        self._lock = Lock()
        self._processed_count = 0
        self._failed_count = 0
    
    def put(self, event: Event):
        """Add event to queue."""
        self._queue.put(event)
        self.log.debug(f"Queued event: {event.type}")
    
    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        """Get event from queue."""
        try:
            event = self._queue.get(timeout=timeout)
            with self._lock:
                self._pending[event.id] = event
            return event
        except Empty:
            return None
    
    def ack(self, event_id: str):
        """Acknowledge event processing."""
        with self._lock:
            if event_id in self._pending:
                del self._pending[event_id]
                self._processed_count += 1
                self.log.debug(f"Acknowledged event: {event_id}")
    
    def nack(self, event_id: str, requeue: bool = True):
        """Negative acknowledge - event processing failed."""
        with self._lock:
            if event_id in self._pending:
                event = self._pending.pop(event_id)
                self._failed_count += 1
                if requeue:
                    self._queue.put(event)
                    self.log.debug(f"Requeued event: {event_id}")
    
    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    def pending_count(self) -> int:
        """Get count of pending events."""
        with self._lock:
            return len(self._pending)
    
    def stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "queued": self._queue.qsize(),
            "pending": self.pending_count(),
            "processed": self._processed_count,
            "failed": self._failed_count
        }


# Global event bus
event_bus = EventEmitter("global")


# Convenience functions
def emit(
    event_type: Union[EventType, str],
    data: Optional[Dict[str, Any]] = None,
    source: str = "unknown"
) -> Event:
    """Emit event on global bus."""
    return event_bus.emit(event_type, data, source)


async def emit_async(
    event_type: Union[EventType, str],
    data: Optional[Dict[str, Any]] = None,
    source: str = "unknown"
) -> Event:
    """Emit event asynchronously on global bus."""
    return await event_bus.emit_async(event_type, data, source)


def on(
    event_type: Union[EventType, str],
    callback: Callable,
    priority: EventPriority = EventPriority.NORMAL
) -> str:
    """Register handler on global bus."""
    return event_bus.on(event_type, callback, priority)


def off(event_type: Union[EventType, str], handler_id: str) -> bool:
    """Remove handler from global bus."""
    return event_bus.off(event_type, handler_id)
