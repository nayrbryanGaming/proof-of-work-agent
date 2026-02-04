"""
Circuit Breaker pattern implementation for fault tolerance.
Prevents cascading failures by stopping operations when failure threshold is reached.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps

from agent.logger import get_logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitBreakerError(Exception):
    """Error raised when circuit is open."""
    
    def __init__(self, name: str, state: CircuitState, retry_after: Optional[float] = None):
        self.name = name
        self.state = state
        self.retry_after = retry_after
        message = f"Circuit breaker '{name}' is {state.value}"
        if retry_after:
            message += f", retry after {retry_after:.1f}s"
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker implementation with configurable thresholds.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Blocking all calls, waiting for reset timeout
    - HALF_OPEN: Testing with limited calls to see if service recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        exclude_exceptions: Optional[tuple] = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            failure_threshold: Number of consecutive failures before opening
            success_threshold: Number of consecutive successes to close from half-open
            reset_timeout: Seconds to wait before transitioning from open to half-open
            half_open_max_calls: Max concurrent calls in half-open state
            exclude_exceptions: Exception types that should not count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exclude_exceptions = exclude_exceptions or ()
        
        self.log = get_logger(f"circuit.{name}")
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN
    
    def _should_try_reset(self) -> bool:
        """Check if enough time has passed to try resetting."""
        if self.state != CircuitState.OPEN:
            return False
        return time.time() - self._state_changed_at >= self.reset_timeout
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self._state_changed_at = time.time()
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        
        self.log.info(f"State transition: {old_state.value} -> {new_state.value}")
    
    def _record_success(self):
        """Record a successful call."""
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.last_success_time = time.time()
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_calls -= 1
            if self.stats.consecutive_successes >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self, error: Exception):
        """Record a failed call."""
        self.stats.total_calls += 1
        self.stats.failed_calls += 1
        self.stats.last_failure_time = time.time()
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_calls -= 1
            self._transition_to(CircuitState.OPEN)
        elif self.state == CircuitState.CLOSED:
            if self.stats.consecutive_failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    async def _acquire(self) -> bool:
        """Acquire permission to make a call."""
        async with self._lock:
            # Check for state transitions
            if self.state == CircuitState.OPEN:
                if self._should_try_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self.stats.rejected_calls += 1
                    retry_after = self.reset_timeout - (time.time() - self._state_changed_at)
                    raise CircuitBreakerError(self.name, self.state, retry_after)
            
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self.stats.rejected_calls += 1
                    raise CircuitBreakerError(self.name, self.state)
                self._half_open_calls += 1
            
            return True
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async or sync function to call
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Result of the function call
        
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function raises (and circuit may open)
        """
        await self._acquire()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except self.exclude_exceptions:
            # Don't count excluded exceptions as failures
            self._record_success()
            raise
            
        except Exception as e:
            self._record_failure(e)
            raise
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.state = CircuitState.CLOSED
        self._state_changed_at = time.time()
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0
        self._half_open_calls = 0
        self.log.info("Circuit breaker manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'stats': {
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'rejected_calls': self.stats.rejected_calls,
                'failure_rate': f"{self.stats.failure_rate:.1%}",
                'consecutive_failures': self.stats.consecutive_failures,
                'consecutive_successes': self.stats.consecutive_successes,
            },
            'config': {
                'failure_threshold': self.failure_threshold,
                'success_threshold': self.success_threshold,
                'reset_timeout': self.reset_timeout,
            }
        }


# ============================================================
# Circuit Breaker Registry
# ============================================================

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.log = get_logger("circuit.registry")
    
    def get_or_create(
        self,
        name: str,
        **kwargs
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, **kwargs)
            self.log.info(f"Created circuit breaker: {name}")
        return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }


# Global registry
registry = CircuitBreakerRegistry()


# ============================================================
# Decorator
# ============================================================

T = TypeVar('T')


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    reset_timeout: float = 60.0,
    exclude_exceptions: Optional[tuple] = None,
):
    """
    Decorator to wrap a function with circuit breaker.
    
    Usage:
        @circuit_breaker(name="api", failure_threshold=3)
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = registry.get_or_create(
            breaker_name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            reset_timeout=reset_timeout,
            exclude_exceptions=exclude_exceptions,
        )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(breaker.call(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
