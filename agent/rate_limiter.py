"""
Rate Limiter - Token bucket and sliding window rate limiting.
Prevents API abuse and ensures fair resource usage.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps
from threading import Lock

from agent.logger import get_logger


T = TypeVar('T')


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_at: Optional[float] = None
    retry_after: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "retry_after": self.retry_after
        }


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, result: RateLimitResult, message: str = "Rate limit exceeded"):
        super().__init__(message)
        self.result = result
        self.retry_after = result.retry_after


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Attempt to acquire tokens."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the rate limiter."""
        pass


class TokenBucket(RateLimiter):
    """
    Token bucket rate limiter.
    
    Tokens are added at a constant rate and can burst up to capacity.
    """
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # Tokens per second
        initial_tokens: Optional[int] = None
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self._last_refill = time.time()
        self._lock = Lock()
        self.log = get_logger("rate_limiter.token_bucket")
    
    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now
    
    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Acquire tokens from the bucket."""
        with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self._tokens),
                    reset_at=None
                )
            else:
                # Calculate when tokens will be available
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.refill_rate
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    retry_after=wait_time
                )
    
    async def acquire_async(self, tokens: int = 1, wait: bool = True) -> RateLimitResult:
        """Acquire tokens, optionally waiting."""
        result = self.acquire(tokens)
        
        if not result.allowed and wait and result.retry_after:
            self.log.debug(f"Waiting {result.retry_after:.2f}s for tokens")
            await asyncio.sleep(result.retry_after)
            result = self.acquire(tokens)
        
        return result
    
    @property
    def tokens(self) -> float:
        """Get current token count."""
        with self._lock:
            self._refill()
            return self._tokens
    
    def reset(self):
        """Reset to full capacity."""
        with self._lock:
            self._tokens = float(self.capacity)
            self._last_refill = time.time()


class SlidingWindowLog(RateLimiter):
    """
    Sliding window log rate limiter.
    
    Tracks each request timestamp and counts within the window.
    More accurate but uses more memory.
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: float
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: deque = deque()
        self._lock = Lock()
    
    def _cleanup(self):
        """Remove old requests outside the window."""
        cutoff = time.time() - self.window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
    
    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed."""
        with self._lock:
            self._cleanup()
            now = time.time()
            
            if len(self._requests) + tokens <= self.max_requests:
                for _ in range(tokens):
                    self._requests.append(now)
                
                return RateLimitResult(
                    allowed=True,
                    remaining=self.max_requests - len(self._requests),
                    reset_at=now + self.window_seconds
                )
            else:
                # Calculate when oldest request will expire
                if self._requests:
                    retry_after = self._requests[0] + self.window_seconds - now
                else:
                    retry_after = 0
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=now + retry_after,
                    retry_after=max(0, retry_after)
                )
    
    def reset(self):
        """Clear all requests."""
        with self._lock:
            self._requests.clear()


class FixedWindowCounter(RateLimiter):
    """
    Fixed window counter rate limiter.
    
    Simple counter that resets at fixed intervals.
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: float
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._count = 0
        self._window_start = time.time()
        self._lock = Lock()
    
    def _maybe_reset_window(self):
        """Reset counter if window has passed."""
        now = time.time()
        if now - self._window_start >= self.window_seconds:
            self._count = 0
            self._window_start = now
    
    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed."""
        with self._lock:
            self._maybe_reset_window()
            
            if self._count + tokens <= self.max_requests:
                self._count += tokens
                
                return RateLimitResult(
                    allowed=True,
                    remaining=self.max_requests - self._count,
                    reset_at=self._window_start + self.window_seconds
                )
            else:
                retry_after = (self._window_start + self.window_seconds) - time.time()
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=self._window_start + self.window_seconds,
                    retry_after=max(0, retry_after)
                )
    
    def reset(self):
        """Reset counter."""
        with self._lock:
            self._count = 0
            self._window_start = time.time()


class LeakyBucket(RateLimiter):
    """
    Leaky bucket rate limiter.
    
    Requests queue up and are processed at a fixed rate.
    Excess requests are dropped.
    """
    
    def __init__(
        self,
        capacity: int,
        leak_rate: float  # Requests per second
    ):
        self.capacity = capacity
        self.leak_rate = leak_rate
        self._queue = 0.0
        self._last_leak = time.time()
        self._lock = Lock()
    
    def _leak(self):
        """Process queued requests."""
        now = time.time()
        elapsed = now - self._last_leak
        leaked = elapsed * self.leak_rate
        
        self._queue = max(0, self._queue - leaked)
        self._last_leak = now
    
    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Add request to queue."""
        with self._lock:
            self._leak()
            
            if self._queue + tokens <= self.capacity:
                self._queue += tokens
                
                # Wait time is position in queue / rate
                wait_time = self._queue / self.leak_rate
                
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self.capacity - self._queue),
                    retry_after=wait_time
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    retry_after=self._queue / self.leak_rate
                )
    
    def reset(self):
        """Clear queue."""
        with self._lock:
            self._queue = 0
            self._last_leak = time.time()


class CompositeRateLimiter:
    """
    Combines multiple rate limiters.
    All must allow for request to proceed.
    """
    
    def __init__(self, limiters: list[RateLimiter]):
        self.limiters = limiters
    
    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Check all limiters."""
        results = [l.acquire(tokens) for l in self.limiters]
        
        # If any denied, return the one with longest wait
        denied = [r for r in results if not r.allowed]
        if denied:
            worst = max(denied, key=lambda r: r.retry_after or 0)
            return worst
        
        # All allowed, return minimum remaining
        return RateLimitResult(
            allowed=True,
            remaining=min(r.remaining for r in results)
        )
    
    def reset(self):
        """Reset all limiters."""
        for limiter in self.limiters:
            limiter.reset()


class RateLimiterRegistry:
    """Registry for named rate limiters."""
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = Lock()
    
    def register(self, name: str, limiter: RateLimiter):
        """Register a rate limiter."""
        with self._lock:
            self._limiters[name] = limiter
    
    def get(self, name: str) -> Optional[RateLimiter]:
        """Get a rate limiter by name."""
        with self._lock:
            return self._limiters.get(name)
    
    def create_token_bucket(
        self,
        name: str,
        capacity: int,
        refill_rate: float
    ) -> TokenBucket:
        """Create and register a token bucket limiter."""
        limiter = TokenBucket(capacity, refill_rate)
        self.register(name, limiter)
        return limiter
    
    def create_sliding_window(
        self,
        name: str,
        max_requests: int,
        window_seconds: float
    ) -> SlidingWindowLog:
        """Create and register a sliding window limiter."""
        limiter = SlidingWindowLog(max_requests, window_seconds)
        self.register(name, limiter)
        return limiter


# Global registry
rate_limiter_registry = RateLimiterRegistry()


# Pre-configured limiters for common APIs
colosseum_limiter = TokenBucket(capacity=60, refill_rate=1)  # 60 requests per minute
openai_limiter = TokenBucket(capacity=50, refill_rate=0.5)   # 30 requests per minute
solana_limiter = TokenBucket(capacity=100, refill_rate=2)    # 120 requests per minute


def rate_limit(
    limiter: Optional[RateLimiter] = None,
    limiter_name: Optional[str] = None,
    tokens: int = 1,
    wait: bool = False,
    raise_on_limit: bool = True
):
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        limiter: Rate limiter instance
        limiter_name: Name of registered rate limiter
        tokens: Number of tokens to acquire
        wait: Whether to wait if rate limited
        raise_on_limit: Whether to raise exception on limit
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            rl = limiter or rate_limiter_registry.get(limiter_name or "")
            if not rl:
                return await func(*args, **kwargs)
            
            if wait and hasattr(rl, 'acquire_async'):
                result = await rl.acquire_async(tokens, wait=True)
            else:
                result = rl.acquire(tokens)
            
            if not result.allowed:
                if raise_on_limit:
                    raise RateLimitExceeded(result)
                return None
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            rl = limiter or rate_limiter_registry.get(limiter_name or "")
            if not rl:
                return func(*args, **kwargs)
            
            result = rl.acquire(tokens)
            
            if not result.allowed:
                if raise_on_limit:
                    raise RateLimitExceeded(result)
                return None
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class AdaptiveRateLimiter:
    """
    Rate limiter that adapts based on response headers.
    Commonly used with APIs that return rate limit info.
    """
    
    def __init__(
        self,
        initial_capacity: int = 100,
        min_capacity: int = 10
    ):
        self.min_capacity = min_capacity
        self._capacity = initial_capacity
        self._remaining = initial_capacity
        self._reset_at: Optional[float] = None
        self._lock = Lock()
        self.log = get_logger("rate_limiter.adaptive")
    
    def update_from_headers(self, headers: Dict[str, str]):
        """Update limits from response headers."""
        with self._lock:
            # Common header names
            limit = headers.get('X-RateLimit-Limit') or headers.get('x-ratelimit-limit')
            remaining = headers.get('X-RateLimit-Remaining') or headers.get('x-ratelimit-remaining')
            reset = headers.get('X-RateLimit-Reset') or headers.get('x-ratelimit-reset')
            
            if limit:
                self._capacity = int(limit)
            if remaining:
                self._remaining = int(remaining)
            if reset:
                self._reset_at = float(reset)
            
            self.log.debug(
                f"Updated limits: capacity={self._capacity}, "
                f"remaining={self._remaining}"
            )
    
    def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed."""
        with self._lock:
            now = time.time()
            
            # Reset if needed
            if self._reset_at and now >= self._reset_at:
                self._remaining = self._capacity
                self._reset_at = None
            
            if self._remaining >= tokens:
                self._remaining -= tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=self._remaining,
                    reset_at=self._reset_at
                )
            else:
                retry_after = (self._reset_at - now) if self._reset_at else 60.0
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=self._reset_at,
                    retry_after=max(0, retry_after)
                )
    
    def reset(self):
        """Reset to initial state."""
        with self._lock:
            self._remaining = self._capacity
            self._reset_at = None
