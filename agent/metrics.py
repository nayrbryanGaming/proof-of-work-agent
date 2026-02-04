"""
Advanced Metrics System with Prometheus-compatible export.
Real-time metrics collection, aggregation, and reporting.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from agent.logger import get_logger


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"       # Monotonically increasing
    GAUGE = "gauge"           # Can go up or down
    HISTOGRAM = "histogram"   # Distribution of values
    SUMMARY = "summary"       # Statistical summary


@dataclass
class MetricLabel:
    """Label for metric identification."""
    name: str
    value: str
    
    def __hash__(self):
        return hash((self.name, self.value))
    
    def __eq__(self, other):
        if not isinstance(other, MetricLabel):
            return False
        return self.name == other.name and self.value == other.value


@dataclass
class MetricSample:
    """Single metric sample."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """Histogram bucket for distribution tracking."""
    upper_bound: float
    count: int = 0


class Counter:
    """Monotonically increasing counter metric."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = Lock()
        self._created_at = time.time()
    
    def inc(self, value: float = 1.0):
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented with positive values")
        with self._lock:
            self._value += value
    
    @property
    def value(self) -> float:
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset counter (use with caution)."""
        with self._lock:
            self._value = 0.0
    
    def collect(self) -> MetricSample:
        return MetricSample(
            name=self.name,
            value=self.value,
            timestamp=time.time(),
            labels=self.labels,
            metric_type=MetricType.COUNTER
        )


class Gauge:
    """Gauge metric that can go up or down."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = Lock()
    
    def set(self, value: float):
        """Set gauge value."""
        with self._lock:
            self._value = value
    
    def inc(self, value: float = 1.0):
        """Increment gauge."""
        with self._lock:
            self._value += value
    
    def dec(self, value: float = 1.0):
        """Decrement gauge."""
        with self._lock:
            self._value -= value
    
    @property
    def value(self) -> float:
        with self._lock:
            return self._value
    
    def set_to_current_time(self):
        """Set gauge to current Unix timestamp."""
        self.set(time.time())
    
    def collect(self) -> MetricSample:
        return MetricSample(
            name=self.name,
            value=self.value,
            timestamp=time.time(),
            labels=self.labels,
            metric_type=MetricType.GAUGE
        )


class Histogram:
    """Histogram metric for distribution tracking."""
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[tuple] = None
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._bucket_counts = {b: 0 for b in self._buckets}
        self._sum = 0.0
        self._count = 0
        self._lock = Lock()
    
    def observe(self, value: float):
        """Observe a value."""
        with self._lock:
            self._sum += value
            self._count += 1
            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
    
    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum
    
    @property
    def count(self) -> int:
        with self._lock:
            return self._count
    
    @property
    def mean(self) -> float:
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._sum / self._count
    
    def get_buckets(self) -> Dict[float, int]:
        with self._lock:
            return dict(self._bucket_counts)
    
    def collect(self) -> List[MetricSample]:
        """Collect all histogram samples."""
        samples = []
        with self._lock:
            for bucket, count in self._bucket_counts.items():
                samples.append(MetricSample(
                    name=f"{self.name}_bucket",
                    value=count,
                    timestamp=time.time(),
                    labels={**self.labels, "le": str(bucket)},
                    metric_type=MetricType.HISTOGRAM
                ))
            samples.append(MetricSample(
                name=f"{self.name}_sum",
                value=self._sum,
                timestamp=time.time(),
                labels=self.labels,
                metric_type=MetricType.HISTOGRAM
            ))
            samples.append(MetricSample(
                name=f"{self.name}_count",
                value=self._count,
                timestamp=time.time(),
                labels=self.labels,
                metric_type=MetricType.HISTOGRAM
            ))
        return samples


class Summary:
    """Summary metric with quantile calculation."""
    
    DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99)
    MAX_AGE = 600  # 10 minutes
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        quantiles: Optional[tuple] = None,
        max_age: float = 600
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._quantiles = quantiles or self.DEFAULT_QUANTILES
        self._values: deque = deque()
        self._sum = 0.0
        self._count = 0
        self._max_age = max_age
        self._lock = Lock()
    
    def observe(self, value: float):
        """Observe a value."""
        now = time.time()
        with self._lock:
            self._values.append((now, value))
            self._sum += value
            self._count += 1
            self._cleanup()
    
    def _cleanup(self):
        """Remove old values."""
        cutoff = time.time() - self._max_age
        while self._values and self._values[0][0] < cutoff:
            self._values.popleft()
    
    def get_quantile(self, q: float) -> float:
        """Calculate quantile from recent values."""
        with self._lock:
            self._cleanup()
            if not self._values:
                return 0.0
            
            values = sorted(v for _, v in self._values)
            idx = int(q * len(values))
            idx = min(idx, len(values) - 1)
            return values[idx]
    
    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum
    
    @property
    def count(self) -> int:
        with self._lock:
            return self._count
    
    def collect(self) -> List[MetricSample]:
        """Collect all summary samples."""
        samples = []
        for q in self._quantiles:
            samples.append(MetricSample(
                name=self.name,
                value=self.get_quantile(q),
                timestamp=time.time(),
                labels={**self.labels, "quantile": str(q)},
                metric_type=MetricType.SUMMARY
            ))
        with self._lock:
            samples.append(MetricSample(
                name=f"{self.name}_sum",
                value=self._sum,
                timestamp=time.time(),
                labels=self.labels,
                metric_type=MetricType.SUMMARY
            ))
            samples.append(MetricSample(
                name=f"{self.name}_count",
                value=self._count,
                timestamp=time.time(),
                labels=self.labels,
                metric_type=MetricType.SUMMARY
            ))
        return samples


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, histogram: Histogram):
        self._histogram = histogram
        self._start: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            self._histogram.observe(time.time() - self._start)


class MetricsRegistry:
    """Registry for all metrics."""
    
    _instance: Optional["MetricsRegistry"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.log = get_logger("metrics")
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary]] = {}
        self._lock = Lock()
        self._initialized = True
    
    def register(self, metric: Union[Counter, Gauge, Histogram, Summary]):
        """Register a metric."""
        with self._lock:
            if metric.name in self._metrics:
                self.log.warn(f"Metric {metric.name} already registered, overwriting")
            self._metrics[metric.name] = metric
    
    def unregister(self, name: str):
        """Unregister a metric."""
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
    
    def get(self, name: str) -> Optional[Union[Counter, Gauge, Histogram, Summary]]:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)
    
    def counter(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Counter:
        """Create and register a counter."""
        counter = Counter(name, description, labels)
        self.register(counter)
        return counter
    
    def gauge(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Create and register a gauge."""
        gauge = Gauge(name, description, labels)
        self.register(gauge)
        return gauge
    
    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[tuple] = None
    ) -> Histogram:
        """Create and register a histogram."""
        histogram = Histogram(name, description, labels, buckets)
        self.register(histogram)
        return histogram
    
    def summary(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        quantiles: Optional[tuple] = None
    ) -> Summary:
        """Create and register a summary."""
        summary = Summary(name, description, labels, quantiles)
        self.register(summary)
        return summary
    
    def collect_all(self) -> List[MetricSample]:
        """Collect samples from all registered metrics."""
        samples = []
        with self._lock:
            for metric in self._metrics.values():
                result = metric.collect()
                if isinstance(result, list):
                    samples.extend(result)
                else:
                    samples.append(result)
        return samples
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        samples = self.collect_all()
        
        for sample in samples:
            labels_str = ""
            if sample.labels:
                labels_parts = [f'{k}="{v}"' for k, v in sample.labels.items()]
                labels_str = "{" + ",".join(labels_parts) + "}"
            
            lines.append(f"{sample.name}{labels_str} {sample.value}")
        
        return "\n".join(lines)
    
    def to_json(self) -> Dict[str, Any]:
        """Export metrics as JSON."""
        result = {}
        with self._lock:
            for name, metric in self._metrics.items():
                if isinstance(metric, Counter):
                    result[name] = {"type": "counter", "value": metric.value, "labels": metric.labels}
                elif isinstance(metric, Gauge):
                    result[name] = {"type": "gauge", "value": metric.value, "labels": metric.labels}
                elif isinstance(metric, Histogram):
                    result[name] = {
                        "type": "histogram",
                        "sum": metric.sum,
                        "count": metric.count,
                        "mean": metric.mean,
                        "buckets": metric.get_buckets(),
                        "labels": metric.labels
                    }
                elif isinstance(metric, Summary):
                    quantiles = {str(q): metric.get_quantile(q) for q in metric._quantiles}
                    result[name] = {
                        "type": "summary",
                        "sum": metric.sum,
                        "count": metric.count,
                        "quantiles": quantiles,
                        "labels": metric.labels
                    }
        return result


# Global registry
registry = MetricsRegistry()


# Agent-specific metrics
class AgentMetricsCollector:
    """Collector for agent-specific metrics."""
    
    def __init__(self):
        self.log = get_logger("metrics.agent")
        
        # Counters
        self.cycles_total = registry.counter(
            "pow_agent_cycles_total",
            "Total number of cycles executed"
        )
        self.cycles_success = registry.counter(
            "pow_agent_cycles_success_total",
            "Total number of successful cycles"
        )
        self.cycles_failed = registry.counter(
            "pow_agent_cycles_failed_total",
            "Total number of failed cycles"
        )
        self.tasks_solved = registry.counter(
            "pow_agent_tasks_solved_total",
            "Total number of tasks solved"
        )
        self.forum_votes = registry.counter(
            "pow_agent_forum_votes_total",
            "Total number of forum votes"
        )
        self.forum_comments = registry.counter(
            "pow_agent_forum_comments_total",
            "Total number of forum comments"
        )
        self.solana_transactions = registry.counter(
            "pow_agent_solana_transactions_total",
            "Total number of Solana transactions"
        )
        self.errors_total = registry.counter(
            "pow_agent_errors_total",
            "Total number of errors"
        )
        
        # Gauges
        self.current_cycle = registry.gauge(
            "pow_agent_current_cycle",
            "Current cycle number"
        )
        self.agent_running = registry.gauge(
            "pow_agent_running",
            "Whether agent is running (1=yes, 0=no)"
        )
        self.last_heartbeat_timestamp = registry.gauge(
            "pow_agent_last_heartbeat_timestamp",
            "Unix timestamp of last heartbeat check"
        )
        self.uptime_seconds = registry.gauge(
            "pow_agent_uptime_seconds",
            "Agent uptime in seconds"
        )
        
        # Histograms
        self.cycle_duration = registry.histogram(
            "pow_agent_cycle_duration_seconds",
            "Duration of each cycle in seconds",
            buckets=(1, 5, 10, 30, 60, 120, 300, 600)
        )
        self.task_solve_duration = registry.histogram(
            "pow_agent_task_solve_duration_seconds",
            "Duration to solve tasks in seconds",
            buckets=(1, 2, 5, 10, 30, 60)
        )
        self.api_request_duration = registry.histogram(
            "pow_agent_api_request_duration_seconds",
            "Duration of API requests in seconds",
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10)
        )
        
        self._start_time = time.time()
    
    def record_cycle_start(self, cycle_num: int):
        """Record start of a cycle."""
        self.current_cycle.set(cycle_num)
        self.cycles_total.inc()
    
    def record_cycle_complete(self, duration: float, success: bool):
        """Record completion of a cycle."""
        self.cycle_duration.observe(duration)
        if success:
            self.cycles_success.inc()
        else:
            self.cycles_failed.inc()
    
    def record_task_solved(self, duration: float):
        """Record a solved task."""
        self.tasks_solved.inc()
        self.task_solve_duration.observe(duration)
    
    def record_forum_activity(self, votes: int = 0, comments: int = 0):
        """Record forum activity."""
        if votes > 0:
            self.forum_votes.inc(votes)
        if comments > 0:
            self.forum_comments.inc(comments)
    
    def record_solana_tx(self):
        """Record a Solana transaction."""
        self.solana_transactions.inc()
    
    def record_error(self):
        """Record an error."""
        self.errors_total.inc()
    
    def record_heartbeat(self):
        """Record a heartbeat check."""
        self.last_heartbeat_timestamp.set_to_current_time()
    
    def set_running(self, running: bool):
        """Set agent running status."""
        self.agent_running.set(1 if running else 0)
    
    def update_uptime(self):
        """Update uptime metric."""
        self.uptime_seconds.set(time.time() - self._start_time)
    
    def time_api_request(self) -> Timer:
        """Get a timer for API requests."""
        return Timer(self.api_request_duration)
    
    def time_cycle(self) -> Timer:
        """Get a timer for cycles."""
        return Timer(self.cycle_duration)
    
    def time_task(self) -> Timer:
        """Get a timer for task solving."""
        return Timer(self.task_solve_duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "cycles": {
                "total": self.cycles_total.value,
                "success": self.cycles_success.value,
                "failed": self.cycles_failed.value,
                "current": self.current_cycle.value
            },
            "tasks": {
                "solved": self.tasks_solved.value,
                "avg_duration": self.task_solve_duration.mean
            },
            "forum": {
                "votes": self.forum_votes.value,
                "comments": self.forum_comments.value
            },
            "solana": {
                "transactions": self.solana_transactions.value
            },
            "health": {
                "running": bool(self.agent_running.value),
                "uptime_seconds": time.time() - self._start_time,
                "errors": self.errors_total.value,
                "last_heartbeat": self.last_heartbeat_timestamp.value
            }
        }


# Global metrics collector
agent_metrics = AgentMetricsCollector()
