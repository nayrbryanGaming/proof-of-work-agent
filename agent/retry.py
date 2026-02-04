"""
Retry Strategies - Advanced retry patterns with multiple policies.
Supports exponential backoff, jitter, circuit breaker integration.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from agent.logger import get_logger


T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    DECORRELATED_JITTER = "decorrelated_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    exclude_exceptions: Tuple[Type[Exception], ...] = ()
    on_retry: Optional[Callable[[Exception, int], None]] = None
    

class BackoffPolicy(ABC):
    """Abstract backoff policy."""
    
    @abstractmethod
    def get_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        """Calculate delay for given attempt."""
        pass


class FixedBackoff(BackoffPolicy):
    """Fixed delay between retries."""
    
    def get_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        return min(base_delay, max_delay)


class ExponentialBackoff(BackoffPolicy):
    """Exponential backoff with optional jitter."""
    
    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier
    
    def get_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        delay = base_delay * (self.multiplier ** attempt)
        return min(delay, max_delay)


class LinearBackoff(BackoffPolicy):
    """Linear increase in delay."""
    
    def __init__(self, increment: float = 1.0):
        self.increment = increment
    
    def get_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        delay = base_delay + (self.increment * attempt)
        return min(delay, max_delay)


class FibonacciBackoff(BackoffPolicy):
    """Fibonacci sequence for delay calculation."""
    
    def __init__(self):
        self._cache: Dict[int, int] = {0: 0, 1: 1}
    
    def _fib(self, n: int) -> int:
        if n in self._cache:
            return self._cache[n]
        self._cache[n] = self._fib(n - 1) + self._fib(n - 2)
        return self._cache[n]
    
    def get_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        delay = base_delay * self._fib(attempt + 1)
        return min(delay, max_delay)


class DecorrelatedJitterBackoff(BackoffPolicy):
    """AWS-style decorrelated jitter backoff."""
    
    def __init__(self):
        self._last_delay = 0.0
    
    def get_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        if attempt == 0:
            self._last_delay = base_delay
        else:
            self._last_delay = random.uniform(base_delay, self._last_delay * 3)
        return min(self._last_delay, max_delay)


def _get_backoff_policy(strategy: RetryStrategy) -> BackoffPolicy:
    """Get backoff policy for strategy."""
    policies = {
        RetryStrategy.FIXED: FixedBackoff(),
        RetryStrategy.EXPONENTIAL: ExponentialBackoff(),
        RetryStrategy.LINEAR: LinearBackoff(),
        RetryStrategy.FIBONACCI: FibonacciBackoff(),
        RetryStrategy.DECORRELATED_JITTER: DecorrelatedJitterBackoff(),
    }
    return policies.get(strategy, ExponentialBackoff())


def _add_jitter(delay: float, factor: float) -> float:
    """Add jitter to delay."""
    jitter_range = delay * factor
    return delay + random.uniform(-jitter_range, jitter_range)


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0
    history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class Retrier:
    """
    Advanced retry mechanism with configurable strategies.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.log = get_logger("retry")
        self._policy = _get_backoff_policy(self.config.strategy)
    
    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry."""
        if isinstance(exception, self.config.exclude_exceptions):
            return False
        return isinstance(exception, self.config.retry_exceptions)
    
    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt."""
        delay = self._policy.get_delay(
            attempt,
            self.config.base_delay,
            self.config.max_delay
        )
        
        if self.config.jitter:
            delay = _add_jitter(delay, self.config.jitter_factor)
        
        return max(0, delay)
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> RetryResult:
        """Execute function with retry logic (sync)."""
        history = []
        total_delay = 0.0
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                history.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "duration": time.time() - start_time
                })
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_delay=total_delay,
                    history=history
                )
                
            except Exception as e:
                last_exception = e
                duration = time.time() - start_time
                
                history.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": duration
                })
                
                if not self._should_retry(e):
                    self.log.warn(f"Non-retryable exception: {type(e).__name__}")
                    break
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._get_delay(attempt)
                    total_delay += delay
                    
                    self.log.info(
                        f"Attempt {attempt + 1} failed, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    if self.config.on_retry:
                        self.config.on_retry(e, attempt + 1)
                    
                    time.sleep(delay)
        
        return RetryResult(
            success=False,
            exception=last_exception,
            attempts=len(history),
            total_delay=total_delay,
            history=history
        )
    
    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> RetryResult:
        """Execute async function with retry logic."""
        history = []
        total_delay = 0.0
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                history.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "duration": time.time() - start_time
                })
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_delay=total_delay,
                    history=history
                )
                
            except Exception as e:
                last_exception = e
                duration = time.time() - start_time
                
                history.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": duration
                })
                
                if not self._should_retry(e):
                    break
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._get_delay(attempt)
                    total_delay += delay
                    
                    self.log.info(
                        f"Attempt {attempt + 1} failed, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    if self.config.on_retry:
                        self.config.on_retry(e, attempt + 1)
                    
                    await asyncio.sleep(delay)
        
        return RetryResult(
            success=False,
            exception=last_exception,
            attempts=len(history),
            total_delay=total_delay,
            history=history
        )


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    exclude: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for adding retry logic to functions.
    
    Usage:
        @retry(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL)
        def my_function():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        jitter=jitter,
        retry_exceptions=retry_on,
        exclude_exceptions=exclude,
        on_retry=on_retry
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retrier = Retrier(config)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                result = await retrier.execute_async(func, *args, **kwargs)
                if not result.success:
                    raise result.exception
                return result.result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                result = retrier.execute(func, *args, **kwargs)
                if not result.success:
                    raise result.exception
                return result.result
            return sync_wrapper
    
    return decorator


class RetryBudget:
    """
    Retry budget to limit retries across a time window.
    Prevents retry storms.
    """
    
    def __init__(
        self,
        max_retries_per_window: int = 100,
        window_seconds: float = 60.0,
        min_retries_per_request: int = 1
    ):
        self.max_retries = max_retries_per_window
        self.window = window_seconds
        self.min_retries = min_retries_per_request
        
        self._retry_times: List[float] = []
        self._total_requests = 0
    
    def _cleanup(self):
        """Remove old retries outside window."""
        cutoff = time.time() - self.window
        self._retry_times = [t for t in self._retry_times if t > cutoff]
    
    def can_retry(self) -> bool:
        """Check if retry is allowed within budget."""
        self._cleanup()
        return len(self._retry_times) < self.max_retries
    
    def record_retry(self):
        """Record a retry attempt."""
        self._retry_times.append(time.time())
    
    def get_allowed_retries(self) -> int:
        """Get number of retries allowed for next request."""
        self._cleanup()
        remaining = self.max_retries - len(self._retry_times)
        return max(self.min_retries, remaining)
    
    def stats(self) -> Dict[str, Any]:
        """Get budget statistics."""
        self._cleanup()
        return {
            "retries_in_window": len(self._retry_times),
            "max_retries": self.max_retries,
            "window_seconds": self.window,
            "remaining_budget": self.max_retries - len(self._retry_times)
        }


class AdaptiveRetrier:
    """
    Adaptive retry mechanism that adjusts based on success rates.
    """
    
    def __init__(
        self,
        initial_config: Optional[RetryConfig] = None,
        success_threshold: float = 0.8,
        failure_threshold: float = 0.5,
        adjustment_window: int = 100
    ):
        self.config = initial_config or RetryConfig()
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.adjustment_window = adjustment_window
        
        self.log = get_logger("adaptive_retry")
        self._results: List[bool] = []
        self._current_multiplier = 1.0
    
    def _record_result(self, success: bool):
        """Record result and adjust configuration."""
        self._results.append(success)
        
        if len(self._results) > self.adjustment_window:
            self._results = self._results[-self.adjustment_window:]
        
        if len(self._results) >= 10:
            success_rate = sum(self._results) / len(self._results)
            
            if success_rate >= self.success_threshold:
                # Reduce delays, fewer retries needed
                self._current_multiplier = max(0.5, self._current_multiplier * 0.9)
                self.log.debug(f"High success rate ({success_rate:.2%}), reducing delays")
            elif success_rate <= self.failure_threshold:
                # Increase delays, service is struggling
                self._current_multiplier = min(3.0, self._current_multiplier * 1.2)
                self.log.debug(f"Low success rate ({success_rate:.2%}), increasing delays")
    
    @property
    def adjusted_base_delay(self) -> float:
        """Get adjusted base delay."""
        return self.config.base_delay * self._current_multiplier
    
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> RetryResult:
        """Execute with adaptive retry."""
        adjusted_config = RetryConfig(
            max_attempts=self.config.max_attempts,
            base_delay=self.adjusted_base_delay,
            max_delay=self.config.max_delay,
            strategy=self.config.strategy,
            jitter=self.config.jitter,
            jitter_factor=self.config.jitter_factor,
            retry_exceptions=self.config.retry_exceptions,
            exclude_exceptions=self.config.exclude_exceptions,
            on_retry=self.config.on_retry
        )
        
        retrier = Retrier(adjusted_config)
        result = await retrier.execute_async(func, *args, **kwargs)
        
        self._record_result(result.success)
        
        return result
    
    def stats(self) -> Dict[str, Any]:
        """Get adaptive retrier statistics."""
        success_count = sum(self._results)
        total = len(self._results)
        
        return {
            "success_rate": success_count / total if total > 0 else 0.0,
            "total_attempts": total,
            "current_multiplier": self._current_multiplier,
            "adjusted_base_delay": self.adjusted_base_delay
        }
