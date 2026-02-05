#!/usr/bin/env python3
"""
Watchdog - Self-Healing and Recovery System
Monitors agent health and automatically recovers from failures.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from agent.logger import get_logger


class RecoveryAction(Enum):
    """Actions the watchdog can take."""
    NONE = "none"
    RESTART_COMPONENT = "restart_component"
    CLEAR_STATE = "clear_state"
    RESET_CONNECTIONS = "reset_connections"
    FULL_RESTART = "full_restart"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_fn: Callable[[], bool]
    interval_seconds: float = 60.0
    failure_threshold: int = 3
    recovery_action: RecoveryAction = RecoveryAction.NONE
    last_check: float = 0.0
    consecutive_failures: int = 0
    total_failures: int = 0
    is_critical: bool = False


@dataclass
class WatchdogEvent:
    """Event recorded by the watchdog."""
    timestamp: str
    event_type: str
    component: str
    message: str
    action_taken: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "type": self.event_type,
            "component": self.component,
            "message": self.message,
            "action": self.action_taken
        }


class Watchdog:
    """
    Self-healing watchdog that monitors agent components.
    
    Features:
    - Periodic health checks
    - Automatic recovery actions
    - Event logging
    - Graceful degradation
    """
    
    MAX_EVENTS = 100
    
    def __init__(self):
        self.log = get_logger("watchdog")
        self._checks: Dict[str, HealthCheck] = {}
        self._events: List[WatchdogEvent] = []
        self._recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time = time.time()
        
        # Register default recovery handlers
        self._recovery_handlers[RecoveryAction.CLEAR_STATE] = self._clear_state
        self._recovery_handlers[RecoveryAction.RESET_CONNECTIONS] = self._reset_connections
    
    def register_check(
        self,
        name: str,
        check_fn: Callable[[], bool],
        interval: float = 60.0,
        threshold: int = 3,
        action: RecoveryAction = RecoveryAction.NONE,
        critical: bool = False
    ):
        """Register a health check."""
        self._checks[name] = HealthCheck(
            name=name,
            check_fn=check_fn,
            interval_seconds=interval,
            failure_threshold=threshold,
            recovery_action=action,
            is_critical=critical
        )
        self.log.info(f"Registered health check: {name} (interval: {interval}s)")
    
    def register_recovery_handler(
        self,
        action: RecoveryAction,
        handler: Callable[[], None]
    ):
        """Register a custom recovery handler."""
        self._recovery_handlers[action] = handler
    
    async def start(self):
        """Start the watchdog."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        self.log.info("Watchdog started")
        self._record_event("started", "watchdog", "Watchdog monitoring started")
    
    async def stop(self):
        """Stop the watchdog."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.log.info("Watchdog stopped")
    
    async def _run(self):
        """Main watchdog loop."""
        while self._running:
            try:
                await self._run_checks()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error(f"Watchdog error: {e}")
                await asyncio.sleep(30)
    
    async def _run_checks(self):
        """Run all health checks."""
        now = time.time()
        
        for name, check in self._checks.items():
            # Skip if not time yet
            if now - check.last_check < check.interval_seconds:
                continue
            
            check.last_check = now
            
            try:
                # Run the check
                if asyncio.iscoroutinefunction(check.check_fn):
                    healthy = await check.check_fn()
                else:
                    healthy = check.check_fn()
                
                if healthy:
                    # Reset failure count on success
                    if check.consecutive_failures > 0:
                        self.log.info(f"Health check recovered: {name}")
                        self._record_event("recovered", name, "Health check passed after failures")
                    check.consecutive_failures = 0
                else:
                    self._handle_failure(check)
                    
            except Exception as e:
                self.log.warn(f"Health check exception [{name}]: {e}")
                self._handle_failure(check, str(e))
    
    def _handle_failure(self, check: HealthCheck, error: str = "Check failed"):
        """Handle a health check failure."""
        check.consecutive_failures += 1
        check.total_failures += 1
        
        self.log.warn(f"Health check failed: {check.name} ({check.consecutive_failures}/{check.failure_threshold})")
        
        # Record event
        self._record_event(
            "failure",
            check.name,
            f"Consecutive failures: {check.consecutive_failures}. {error}"
        )
        
        # Check if threshold exceeded
        if check.consecutive_failures >= check.failure_threshold:
            self._trigger_recovery(check)
    
    def _trigger_recovery(self, check: HealthCheck):
        """Trigger recovery action for a failed check."""
        action = check.recovery_action
        
        if action == RecoveryAction.NONE:
            self.log.warn(f"No recovery action for {check.name}")
            return
        
        self.log.warn(f"Triggering recovery action: {action.value} for {check.name}")
        
        handler = self._recovery_handlers.get(action)
        if handler:
            try:
                handler()
                self._record_event("recovery", check.name, f"Recovery action: {action.value}", action.value)
                check.consecutive_failures = 0  # Reset after recovery
            except Exception as e:
                self.log.error(f"Recovery failed for {check.name}: {e}")
                self._record_event("recovery_failed", check.name, str(e))
        else:
            self.log.warn(f"No handler for recovery action: {action.value}")
    
    def _record_event(
        self,
        event_type: str,
        component: str,
        message: str,
        action: Optional[str] = None
    ):
        """Record a watchdog event."""
        event = WatchdogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            component=component,
            message=message,
            action_taken=action
        )
        
        self._events.append(event)
        
        # Trim old events
        if len(self._events) > self.MAX_EVENTS:
            self._events = self._events[-self.MAX_EVENTS:]
    
    def _clear_state(self):
        """Clear agent state files (recovery action)."""
        self.log.info("Clearing state files...")
        state_files = [
            Path("data/state.json"),
            Path("data/commented_posts.json"),
        ]
        for f in state_files:
            if f.exists():
                try:
                    f.unlink()
                    self.log.info(f"Deleted: {f}")
                except Exception as e:
                    self.log.warn(f"Failed to delete {f}: {e}")
    
    def _reset_connections(self):
        """Reset API connections (recovery action)."""
        self.log.info("Resetting connections...")
        # This would reset API clients, clear caches, etc.
        # Implementation depends on how clients are structured
    
    def get_status(self) -> dict:
        """Get watchdog status."""
        checks_status = {}
        for name, check in self._checks.items():
            checks_status[name] = {
                "healthy": check.consecutive_failures == 0,
                "consecutive_failures": check.consecutive_failures,
                "total_failures": check.total_failures,
                "last_check": datetime.fromtimestamp(check.last_check).isoformat() if check.last_check else None,
                "is_critical": check.is_critical
            }
        
        # Overall health
        critical_healthy = all(
            c.consecutive_failures < c.failure_threshold 
            for c in self._checks.values() 
            if c.is_critical
        )
        
        return {
            "running": self._running,
            "uptime": time.time() - self._start_time,
            "overall_healthy": critical_healthy,
            "checks": checks_status,
            "recent_events": [e.to_dict() for e in self._events[-10:]]
        }


# Singleton instance
_watchdog: Optional[Watchdog] = None


def get_watchdog() -> Watchdog:
    """Get the global watchdog instance."""
    global _watchdog
    if _watchdog is None:
        _watchdog = Watchdog()
    return _watchdog
