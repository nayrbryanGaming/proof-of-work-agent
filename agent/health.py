"""
Health Check System - Comprehensive health monitoring.
Monitors all agent components and dependencies.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from threading import Lock

from agent.logger import get_logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "details": self.details
        }


@dataclass
class OverallHealth:
    """Overall system health."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "healthy_count": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
            "total_count": len(self.checks)
        }


class HealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(self, name: str, critical: bool = True, timeout: float = 5.0):
        self.name = name
        self.critical = critical  # If critical, failure makes overall status unhealthy
        self.timeout = timeout
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        pass


class FunctionHealthCheck(HealthCheck):
    """Health check using a custom function."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        critical: bool = True,
        timeout: float = 5.0
    ):
        super().__init__(name, critical, timeout)
        self.check_func = check_func
    
    async def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(
                    self.check_func(),
                    timeout=self.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.check_func),
                    timeout=self.timeout
                )
            
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                message="Check passed" if result else "Check failed",
                latency_ms=latency
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {self.timeout}s",
                latency_ms=self.timeout * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check error: {str(e)}",
                latency_ms=(time.time() - start) * 1000
            )


class HTTPHealthCheck(HealthCheck):
    """Health check via HTTP endpoint."""
    
    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        critical: bool = True,
        timeout: float = 5.0
    ):
        super().__init__(name, critical, timeout)
        self.url = url
        self.expected_status = expected_status
    
    async def check(self) -> HealthCheckResult:
        import aiohttp
        
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, timeout=self.timeout) as resp:
                    latency = (time.time() - start) * 1000
                    
                    if resp.status == self.expected_status:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"HTTP {resp.status}",
                            latency_ms=latency,
                            details={"url": self.url, "status_code": resp.status}
                        )
                    else:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Unexpected status {resp.status}",
                            latency_ms=latency,
                            details={"url": self.url, "status_code": resp.status}
                        )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Request timed out after {self.timeout}s",
                latency_ms=self.timeout * 1000,
                details={"url": self.url}
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Request error: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
                details={"url": self.url, "error": str(e)}
            )


class DiskSpaceHealthCheck(HealthCheck):
    """Health check for disk space."""
    
    def __init__(
        self,
        name: str = "disk_space",
        path: str = "/",
        min_free_percent: float = 10.0,
        critical: bool = True
    ):
        super().__init__(name, critical)
        self.path = path
        self.min_free_percent = min_free_percent
    
    async def check(self) -> HealthCheckResult:
        import shutil
        
        start = time.time()
        try:
            usage = shutil.disk_usage(self.path)
            free_percent = (usage.free / usage.total) * 100
            latency = (time.time() - start) * 1000
            
            status = HealthStatus.HEALTHY if free_percent >= self.min_free_percent else HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=f"{free_percent:.1f}% free space",
                latency_ms=latency,
                details={
                    "path": self.path,
                    "total_gb": usage.total / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "free_percent": free_percent
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check error: {str(e)}",
                latency_ms=(time.time() - start) * 1000
            )


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""
    
    def __init__(
        self,
        name: str = "memory",
        max_percent: float = 90.0,
        critical: bool = True
    ):
        super().__init__(name, critical)
        self.max_percent = max_percent
    
    async def check(self) -> HealthCheckResult:
        import psutil
        
        start = time.time()
        try:
            memory = psutil.virtual_memory()
            latency = (time.time() - start) * 1000
            
            status = HealthStatus.HEALTHY if memory.percent <= self.max_percent else HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=f"{memory.percent:.1f}% memory used",
                latency_ms=latency,
                details={
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent_used": memory.percent
                }
            )
        except ImportError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
                latency_ms=0
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check error: {str(e)}",
                latency_ms=(time.time() - start) * 1000
            )


class ComponentHealthCheck(HealthCheck):
    """Health check for agent components."""
    
    def __init__(
        self,
        name: str,
        component_status_func: Callable[[], Dict[str, Any]],
        required_fields: Optional[List[str]] = None,
        critical: bool = True
    ):
        super().__init__(name, critical)
        self.get_status = component_status_func
        self.required_fields = required_fields or []
    
    async def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self.get_status):
                status = await self.get_status()
            else:
                status = await asyncio.to_thread(self.get_status)
            
            latency = (time.time() - start) * 1000
            
            # Check required fields
            missing = [f for f in self.required_fields if f not in status]
            if missing:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Missing fields: {', '.join(missing)}",
                    latency_ms=latency,
                    details=status
                )
            
            # Check for error indicators
            if status.get("error") or status.get("status") == "error":
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=status.get("error", "Component in error state"),
                    latency_ms=latency,
                    details=status
                )
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Component healthy",
                latency_ms=latency,
                details=status
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check error: {str(e)}",
                latency_ms=(time.time() - start) * 1000
            )


class HealthCheckRegistry:
    """Registry and runner for health checks."""
    
    def __init__(self):
        self.log = get_logger("health")
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = Lock()
        self._start_time = time.time()
        self._last_results: Dict[str, HealthCheckResult] = {}
    
    def register(self, check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self._checks[check.name] = check
            self.log.debug(f"Registered health check: {check.name}")
    
    def unregister(self, name: str):
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        with self._lock:
            check = self._checks.get(name)
        
        if not check:
            return None
        
        try:
            result = await check.check()
            with self._lock:
                self._last_results[name] = result
            return result
        except Exception as e:
            self.log.error(f"Health check {name} failed: {e}")
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check execution failed: {str(e)}"
            )
            with self._lock:
                self._last_results[name] = result
            return result
    
    async def run_all(self, parallel: bool = True) -> OverallHealth:
        """Run all health checks."""
        with self._lock:
            checks = list(self._checks.values())
        
        results: List[HealthCheckResult] = []
        
        if parallel:
            tasks = [check.check() for check in checks]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for check, result in zip(checks, raw_results):
                if isinstance(result, Exception):
                    results.append(HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(result)}"
                    ))
                else:
                    results.append(result)
        else:
            for check in checks:
                try:
                    result = await check.check()
                    results.append(result)
                except Exception as e:
                    results.append(HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(e)}"
                    ))
        
        # Store results
        with self._lock:
            for result in results:
                self._last_results[result.name] = result
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        
        for check, result in zip(checks, results):
            if result.status == HealthStatus.UNHEALTHY:
                if check.critical:
                    overall_status = HealthStatus.UNHEALTHY
                    break
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            elif result.status == HealthStatus.DEGRADED:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        return OverallHealth(
            status=overall_status,
            checks=results,
            uptime_seconds=time.time() - self._start_time
        )
    
    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get last health check results."""
        with self._lock:
            return dict(self._last_results)
    
    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time


# Global registry
health_registry = HealthCheckRegistry()


class HealthMonitor:
    """
    Background health monitoring with alerting.
    """
    
    def __init__(
        self,
        registry: HealthCheckRegistry,
        check_interval: float = 30.0,
        on_status_change: Optional[Callable[[OverallHealth], None]] = None
    ):
        self.registry = registry
        self.check_interval = check_interval
        self.on_status_change = on_status_change
        
        self.log = get_logger("health_monitor")
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_status: Optional[HealthStatus] = None
    
    async def start(self):
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self.log.info(f"Health monitor started (interval: {self.check_interval}s)")
    
    async def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.log.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                health = await self.registry.run_all()
                
                # Check for status change
                if self._last_status != health.status:
                    self.log.info(
                        f"Health status changed: {self._last_status} -> {health.status.value}"
                    )
                    
                    if self.on_status_change:
                        try:
                            if asyncio.iscoroutinefunction(self.on_status_change):
                                await self.on_status_change(health)
                            else:
                                self.on_status_change(health)
                        except Exception as e:
                            self.log.error(f"Status change callback error: {e}")
                    
                    self._last_status = health.status
                
                # Log summary
                healthy = sum(1 for c in health.checks if c.status == HealthStatus.HEALTHY)
                total = len(health.checks)
                self.log.debug(f"Health check: {healthy}/{total} healthy")
                
            except Exception as e:
                self.log.error(f"Monitor loop error: {e}")
            
            await asyncio.sleep(self.check_interval)


# Convenience function for simple checks
def register_health_check(
    name: str,
    check_func: Callable[[], bool],
    critical: bool = True
):
    """Register a simple health check."""
    check = FunctionHealthCheck(name, check_func, critical)
    health_registry.register(check)
    return check
