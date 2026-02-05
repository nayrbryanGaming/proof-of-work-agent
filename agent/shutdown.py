#!/usr/bin/env python3
"""
Graceful Shutdown Handler
Ensures clean shutdown with state persistence and resource cleanup.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import threading

from agent.logger import get_logger


@dataclass
class ShutdownHook:
    """A shutdown hook to be executed during graceful shutdown."""
    name: str
    callback: Callable
    priority: int = 50  # Lower = earlier execution
    timeout: float = 10.0
    is_async: bool = False


class GracefulShutdown:
    """
    Manages graceful shutdown of the agent.
    
    Features:
    - Signal handling (SIGTERM, SIGINT)
    - Ordered shutdown hooks
    - Timeout protection
    - State persistence
    """
    
    SHUTDOWN_STATE_FILE = "data/shutdown_state.json"
    
    def __init__(self):
        self.log = get_logger("shutdown")
        self._hooks: List[ShutdownHook] = []
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._shutdown_event = asyncio.Event()
        self._shutdown_reason: Optional[str] = None
        self._start_time = time.time()
        self._lock = threading.Lock()
        
        # Register default hooks
        self.register("save_shutdown_state", self._save_shutdown_state, priority=100)
    
    def register(
        self,
        name: str,
        callback: Callable,
        priority: int = 50,
        timeout: float = 10.0,
        is_async: bool = False
    ):
        """Register a shutdown hook."""
        hook = ShutdownHook(
            name=name,
            callback=callback,
            priority=priority,
            timeout=timeout,
            is_async=is_async
        )
        
        with self._lock:
            self._hooks.append(hook)
            # Sort by priority
            self._hooks.sort(key=lambda h: h.priority)
        
        self.log.debug(f"Registered shutdown hook: {name} (priority: {priority})")
    
    def unregister(self, name: str):
        """Unregister a shutdown hook."""
        with self._lock:
            self._hooks = [h for h in self._hooks if h.name != name]
    
    def setup_signals(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            self.log.info(f"Received {sig_name}, initiating graceful shutdown...")
            self.request_shutdown(f"Signal: {sig_name}")
        
        # Register signal handlers
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            
            if sys.platform == "win32":
                try:
                    signal.signal(signal.SIGBREAK, signal_handler)
                except (AttributeError, ValueError):
                    pass
        except Exception as e:
            self.log.warn(f"Could not register all signal handlers: {e}")
        
        # Register atexit handler
        atexit.register(self._atexit_handler)
        
        self.log.info("Signal handlers registered")
    
    def request_shutdown(self, reason: str = "Requested"):
        """Request a graceful shutdown."""
        with self._lock:
            if self._shutdown_requested:
                return
            
            self._shutdown_requested = True
            self._shutdown_reason = reason
        
        self.log.info(f"Shutdown requested: {reason}")
        
        # Set async event
        try:
            self._shutdown_event.set()
        except:
            pass
    
    @property
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
    
    async def wait_for_shutdown(self):
        """Wait until shutdown is requested."""
        await self._shutdown_event.wait()
    
    async def execute_shutdown(self) -> bool:
        """Execute all shutdown hooks in order."""
        if self._shutdown_complete:
            return True
        
        self.log.info("=" * 50)
        self.log.info("EXECUTING GRACEFUL SHUTDOWN")
        self.log.info("=" * 50)
        
        start = time.time()
        failed_hooks = []
        
        with self._lock:
            hooks_to_run = self._hooks.copy()
        
        for hook in hooks_to_run:
            self.log.info(f"Running shutdown hook: {hook.name}")
            
            try:
                if hook.is_async:
                    await asyncio.wait_for(
                        hook.callback(),
                        timeout=hook.timeout
                    )
                else:
                    # Run sync callback in executor with timeout
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, hook.callback),
                        timeout=hook.timeout
                    )
                
                self.log.info(f"  ✓ {hook.name} completed")
                
            except asyncio.TimeoutError:
                self.log.warn(f"  ✗ {hook.name} timed out after {hook.timeout}s")
                failed_hooks.append(hook.name)
                
            except Exception as e:
                self.log.error(f"  ✗ {hook.name} failed: {e}")
                failed_hooks.append(hook.name)
        
        duration = time.time() - start
        
        self._shutdown_complete = True
        
        if failed_hooks:
            self.log.warn(f"Shutdown completed with {len(failed_hooks)} failures in {duration:.2f}s")
            return False
        else:
            self.log.info(f"Shutdown completed successfully in {duration:.2f}s")
            return True
    
    def _save_shutdown_state(self):
        """Save shutdown state for debugging and recovery."""
        try:
            Path("data").mkdir(exist_ok=True)
            
            state = {
                "shutdown_at": datetime.now(timezone.utc).isoformat(),
                "reason": self._shutdown_reason,
                "uptime_seconds": time.time() - self._start_time,
                "python_version": sys.version,
            }
            
            Path(self.SHUTDOWN_STATE_FILE).write_text(
                json.dumps(state, indent=2)
            )
        except Exception as e:
            self.log.warn(f"Failed to save shutdown state: {e}")
    
    def _atexit_handler(self):
        """Handler called on interpreter exit."""
        if not self._shutdown_complete:
            self.log.warn("Unexpected exit without graceful shutdown")
            self._save_shutdown_state()
    
    def get_status(self) -> dict:
        """Get shutdown handler status."""
        return {
            "shutdown_requested": self._shutdown_requested,
            "shutdown_complete": self._shutdown_complete,
            "shutdown_reason": self._shutdown_reason,
            "registered_hooks": [h.name for h in self._hooks],
            "uptime_seconds": time.time() - self._start_time
        }


# Singleton instance
_shutdown_handler: Optional[GracefulShutdown] = None


def get_shutdown_handler() -> GracefulShutdown:
    """Get the global shutdown handler."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdown()
    return _shutdown_handler
