"""
Advanced Telemetry System for Proof-of-Work Agent.
Provides comprehensive observability, distributed tracing, and performance monitoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock, local
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from agent.logger import get_logger


T = TypeVar('T')


# ============================================================
# Trace Context
# ============================================================

class TraceContext:
    """
    Distributed trace context for request tracking.
    Compatible with OpenTelemetry/W3C Trace Context.
    """
    
    _local = local()
    
    def __init__(
        self,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        sampled: bool = True,
    ):
        self.trace_id = trace_id or self._generate_trace_id()
        self.span_id = span_id or self._generate_span_id()
        self.parent_span_id = parent_span_id
        self.sampled = sampled
        self.baggage: Dict[str, str] = {}
    
    @staticmethod
    def _generate_trace_id() -> str:
        """Generate 32-char hex trace ID."""
        return uuid.uuid4().hex + uuid.uuid4().hex[:16]
    
    @staticmethod
    def _generate_span_id() -> str:
        """Generate 16-char hex span ID."""
        return uuid.uuid4().hex[:16]
    
    @classmethod
    def current(cls) -> Optional["TraceContext"]:
        """Get current trace context from thread-local storage."""
        return getattr(cls._local, "context", None)
    
    @classmethod
    def set_current(cls, context: Optional["TraceContext"]) -> None:
        """Set current trace context."""
        cls._local.context = context
    
    def create_child(self) -> "TraceContext":
        """Create child context for nested spans."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            sampled=self.sampled,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "sampled": self.sampled,
            "baggage": self.baggage,
        }
    
    def to_header(self) -> str:
        """Convert to W3C traceparent header format."""
        flags = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"
    
    @classmethod
    def from_header(cls, header: str) -> Optional["TraceContext"]:
        """Parse W3C traceparent header."""
        try:
            parts = header.split("-")
            if len(parts) != 4:
                return None
            _, trace_id, span_id, flags = parts
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                sampled=flags == "01",
            )
        except Exception:
            return None


# ============================================================
# Span and Tracing
# ============================================================

class SpanKind(str, Enum):
    """Type of span."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span completion status."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Distributed tracing span."""
    
    trace_id: str
    span_id: str
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def is_ended(self) -> bool:
        return self.end_time is not None
    
    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set span attribute."""
        self.attributes[key] = value
        return self
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add event to span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {}
        ))
        return self
    
    def set_status(self, status: SpanStatus, message: str = "") -> "Span":
        """Set span status."""
        self.status = status
        self.status_message = message
        return self
    
    def end(self, status: Optional[SpanStatus] = None) -> None:
        """End the span."""
        if self.is_ended:
            return
        
        self.end_time = time.time()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind.value,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [asdict(e) for e in self.events],
        }


class Tracer:
    """Distributed tracer for span management."""
    
    MODULE = "tracer"
    
    def __init__(self, service_name: str = "pow-agent"):
        self.log = get_logger(self.MODULE)
        self.service_name = service_name
        self.spans: deque[Span] = deque(maxlen=1000)
        self._lock = Lock()
        self._exporters: List[Callable[[Span], None]] = []
    
    def add_exporter(self, exporter: Callable[[Span], None]) -> None:
        """Add span exporter."""
        self._exporters.append(exporter)
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Start a new span as context manager."""
        context = TraceContext.current()
        
        if context:
            span = Span(
                trace_id=context.trace_id,
                span_id=TraceContext._generate_span_id(),
                parent_span_id=context.span_id,
                name=name,
                kind=kind,
                attributes=attributes or {},
            )
            child_context = context.create_child()
            child_context.span_id = span.span_id
        else:
            new_context = TraceContext()
            span = Span(
                trace_id=new_context.trace_id,
                span_id=new_context.span_id,
                name=name,
                kind=kind,
                attributes=attributes or {},
            )
            child_context = new_context
        
        # Set service name
        span.set_attribute("service.name", self.service_name)
        
        # Store previous context
        previous_context = TraceContext.current()
        TraceContext.set_current(child_context)
        
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {
                "exception.type": type(e).__name__,
                "exception.message": str(e),
            })
            raise
        finally:
            span.end()
            TraceContext.set_current(previous_context)
            self._record_span(span)
    
    def _record_span(self, span: Span) -> None:
        """Record completed span."""
        with self._lock:
            self.spans.append(span)
        
        # Export
        for exporter in self._exporters:
            try:
                exporter(span)
            except Exception as e:
                self.log.warn(f"Span export failed: {e}")
    
    def get_recent_spans(self, count: int = 50) -> List[Span]:
        """Get recent spans."""
        with self._lock:
            return list(self.spans)[-count:]
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return [s for s in self.spans if s.trace_id == trace_id]


# ============================================================
# Performance Profiler
# ============================================================

@dataclass
class ProfileSample:
    """Performance profile sample."""
    
    function: str
    module: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_start: int
    memory_end: int
    memory_delta: int
    cpu_time_ms: float
    tags: Dict[str, str] = field(default_factory=dict)


class Profiler:
    """Performance profiler for function timing."""
    
    MODULE = "profiler"
    
    def __init__(self):
        self.log = get_logger(self.MODULE)
        self.samples: deque[ProfileSample] = deque(maxlen=5000)
        self._lock = Lock()
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    @contextmanager
    def profile(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Profile a code block."""
        if not self._enabled:
            yield
            return
        
        import resource if sys.platform != 'win32' else None
        
        # Start measurements
        start_time = time.time()
        start_cpu = time.process_time()
        
        try:
            start_mem = self._get_memory_usage()
        except:
            start_mem = 0
        
        try:
            yield
        finally:
            # End measurements
            end_time = time.time()
            end_cpu = time.process_time()
            
            try:
                end_mem = self._get_memory_usage()
            except:
                end_mem = 0
            
            sample = ProfileSample(
                function=name,
                module=self._get_caller_module(),
                start_time=start_time,
                end_time=end_time,
                duration_ms=(end_time - start_time) * 1000,
                memory_start=start_mem,
                memory_end=end_mem,
                memory_delta=end_mem - start_mem,
                cpu_time_ms=(end_cpu - start_cpu) * 1000,
                tags=tags or {},
            )
            
            with self._lock:
                self.samples.append(sample)
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except:
            return 0
    
    def _get_caller_module(self) -> str:
        """Get calling module name."""
        import inspect
        for frame_info in inspect.stack():
            if frame_info.filename != __file__:
                return frame_info.filename.split(os.sep)[-1].replace('.py', '')
        return "unknown"
    
    def get_stats(self, function: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling statistics."""
        with self._lock:
            if function:
                samples = [s for s in self.samples if s.function == function]
            else:
                samples = list(self.samples)
        
        if not samples:
            return {}
        
        durations = [s.duration_ms for s in samples]
        
        return {
            "count": len(samples),
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": self._percentile(durations, 50),
            "p95_ms": self._percentile(durations, 95),
            "p99_ms": self._percentile(durations, 99),
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    def get_hot_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get hottest functions by total time."""
        by_function: Dict[str, List[float]] = {}
        
        with self._lock:
            for sample in self.samples:
                if sample.function not in by_function:
                    by_function[sample.function] = []
                by_function[sample.function].append(sample.duration_ms)
        
        results = []
        for func, durations in by_function.items():
            results.append({
                "function": func,
                "call_count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": sum(durations) / len(durations),
            })
        
        results.sort(key=lambda x: x["total_ms"], reverse=True)
        return results[:limit]


# ============================================================
# System Telemetry
# ============================================================

@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    timestamp: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    open_files: int = 0
    threads: int = 0
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemMonitor:
    """System resource monitor."""
    
    MODULE = "system"
    COLLECTION_INTERVAL = 60  # seconds
    
    def __init__(self):
        self.log = get_logger(self.MODULE)
        self.metrics_history: deque[SystemMetrics] = deque(maxlen=1440)  # 24 hours at 1/min
        self._start_time = time.time()
        self._lock = Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def collect(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            proc = psutil.Process()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cpu_percent=cpu,
                memory_percent=mem.percent,
                memory_used_mb=mem.used / (1024 * 1024),
                memory_available_mb=mem.available / (1024 * 1024),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                open_files=len(proc.open_files()),
                threads=proc.num_threads(),
                uptime_seconds=time.time() - self._start_time,
            )
            
        except ImportError:
            # psutil not available
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=time.time() - self._start_time,
            )
        except Exception as e:
            self.log.warn(f"Failed to collect system metrics: {e}")
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        with self._lock:
            self.metrics_history.append(metrics)
        
        return metrics
    
    async def start_collection(self) -> None:
        """Start background metrics collection."""
        if self._running:
            return
        
        self._running = True
        self.log.info("Starting system metrics collection")
        
        while self._running:
            self.collect()
            await asyncio.sleep(self.COLLECTION_INTERVAL)
    
    def stop_collection(self) -> None:
        """Stop background collection."""
        self._running = False
    
    def get_current(self) -> SystemMetrics:
        """Get current metrics."""
        return self.collect()
    
    def get_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history."""
        with self._lock:
            samples_needed = minutes  # 1 sample per minute
            return list(self.metrics_history)[-samples_needed:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            recent = list(self.metrics_history)[-60:]  # Last hour
        
        if not recent:
            return {}
        
        return {
            "current": self.get_current().to_dict(),
            "avg_cpu_percent": sum(m.cpu_percent for m in recent) / len(recent),
            "avg_memory_percent": sum(m.memory_percent for m in recent) / len(recent),
            "max_cpu_percent": max(m.cpu_percent for m in recent),
            "max_memory_percent": max(m.memory_percent for m in recent),
            "uptime_hours": (time.time() - self._start_time) / 3600,
            "samples_count": len(recent),
        }


# ============================================================
# Telemetry Manager
# ============================================================

class TelemetryManager:
    """Central telemetry management."""
    
    MODULE = "telemetry"
    
    def __init__(self, service_name: str = "pow-agent"):
        self.log = get_logger(self.MODULE)
        self.service_name = service_name
        
        # Components
        self.tracer = Tracer(service_name)
        self.profiler = Profiler()
        self.system_monitor = SystemMonitor()
        
        # Output config
        self._output_dir: Optional[Path] = None
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        self.profiler.enabled = value
    
    def configure_output(self, output_dir: Path) -> None:
        """Configure telemetry output directory."""
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        profile: bool = True,
        **attributes
    ):
        """Create a traced and profiled span."""
        if not self._enabled:
            yield None
            return
        
        with self.tracer.start_span(name, kind, attributes) as span:
            if profile:
                with self.profiler.profile(name):
                    yield span
            else:
                yield span
    
    def record_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ) -> None:
        """Record a telemetry event."""
        if not self._enabled:
            return
        
        context = TraceContext.current()
        event = {
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "attributes": attributes or {},
            "trace_id": context.trace_id if context else None,
            "span_id": context.span_id if context else None,
        }
        
        self.log.debug(f"Event: {name}")
        
        if self._output_dir:
            self._write_event(event)
    
    def _write_event(self, event: Dict[str, Any]) -> None:
        """Write event to file."""
        if not self._output_dir:
            return
        
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        file_path = self._output_dir / f"events_{date}.jsonl"
        
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            self.log.warn(f"Failed to write event: {e}")
    
    async def start(self) -> None:
        """Start telemetry collection."""
        if not self._enabled:
            return
        
        self.log.info("Starting telemetry collection")
        asyncio.create_task(self.system_monitor.start_collection())
    
    def stop(self) -> None:
        """Stop telemetry collection."""
        self.system_monitor.stop_collection()
        self.log.info("Telemetry collection stopped")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for telemetry dashboard."""
        return {
            "service_name": self.service_name,
            "enabled": self._enabled,
            "system": self.system_monitor.get_summary(),
            "traces": {
                "recent_count": len(self.tracer.spans),
                "recent_spans": [s.to_dict() for s in self.tracer.get_recent_spans(10)],
            },
            "profiler": {
                "hot_functions": self.profiler.get_hot_functions(),
                "total_samples": len(self.profiler.samples),
            },
        }


# ============================================================
# Global Instance
# ============================================================

_telemetry: Optional[TelemetryManager] = None


def get_telemetry() -> TelemetryManager:
    """Get global telemetry manager."""
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryManager()
    return _telemetry


# ============================================================
# Decorators
# ============================================================

def traced(name: Optional[str] = None, kind: SpanKind = SpanKind.INTERNAL):
    """Decorator to trace a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with get_telemetry().span(span_name, kind):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with get_telemetry().span(span_name, kind):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def profiled(name: Optional[str] = None):
    """Decorator to profile a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        profile_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with get_telemetry().profiler.profile(profile_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with get_telemetry().profiler.profile(profile_name):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator
