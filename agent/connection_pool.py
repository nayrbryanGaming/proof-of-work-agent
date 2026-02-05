"""
Connection Pool & HTTP Client Manager
======================================

Production-grade HTTP client with:
- Connection pooling for efficient reuse
- Automatic retry with exponential backoff
- Request queuing for failed operations
- Circuit breaker integration
- Request/response logging
- Timeout management
- SSL/TLS verification options
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent.logger import get_logger


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_TIMEOUT = 30.0
DEFAULT_POOL_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 2.0
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


# ==============================================================================
# DATA CLASSES
# ==============================================================================

class RequestPriority(Enum):
    """Priority levels for request queue."""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class RequestConfig:
    """Configuration for HTTP requests."""
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5


@dataclass
class RequestResult:
    """Result of an HTTP request."""
    success: bool
    status_code: Optional[int]
    data: Any
    headers: Dict[str, str]
    elapsed_ms: float
    retries: int
    error: Optional[str] = None
    
    def json(self) -> Any:
        """Get response as JSON."""
        if isinstance(self.data, (dict, list)):
            return self.data
        if isinstance(self.data, str):
            return json.loads(self.data)
        return None
    
    def text(self) -> str:
        """Get response as text."""
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, bytes):
            return self.data.decode('utf-8')
        return str(self.data)


@dataclass
class QueuedRequest:
    """Request waiting in retry queue."""
    id: str
    method: str
    url: str
    headers: Dict[str, str]
    data: Any
    priority: RequestPriority
    created_at: float
    retry_count: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    last_error: Optional[str] = None
    callback: Optional[Callable[[RequestResult], None]] = None
    
    def __lt__(self, other):
        # For priority queue ordering
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


# ==============================================================================
# CONNECTION POOL
# ==============================================================================

class ConnectionPool:
    """
    HTTP connection pool with session reuse.
    
    Manages a pool of HTTP sessions for efficient connection reuse,
    reducing latency and system resource usage.
    """
    
    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._sessions: Dict[str, requests.Session] = {}
        self._lock = threading.Lock()
        self._log = get_logger("connection_pool")
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_retries': 0,
            'sessions_created': 0,
        }
    
    def _create_session(self, base_url: str) -> requests.Session:
        """Create a new session with retry configuration."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
        )
        
        # Mount adapter with retry and connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.pool_size,
            pool_maxsize=self.pool_size * 2,
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        self._stats['sessions_created'] += 1
        self._log.info(f"Created new session for {base_url}")
        
        return session
    
    def get_session(self, base_url: str) -> requests.Session:
        """Get or create a session for the given base URL."""
        with self._lock:
            if base_url not in self._sessions:
                self._sessions[base_url] = self._create_session(base_url)
            return self._sessions[base_url]
    
    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        json_data: Any = None,
        params: Optional[Dict[str, str]] = None,
        config: Optional[RequestConfig] = None,
    ) -> RequestResult:
        """
        Make an HTTP request using the connection pool.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
            headers: Optional request headers
            data: Optional form data
            json_data: Optional JSON data
            params: Optional query parameters
            config: Request configuration
        
        Returns:
            RequestResult with response data
        """
        config = config or RequestConfig()
        
        # Extract base URL for session lookup
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        session = self.get_session(base_url)
        start_time = time.time()
        retries = 0
        
        self._stats['total_requests'] += 1
        
        try:
            response = session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                data=data,
                json=json_data,
                params=params,
                timeout=config.timeout,
                verify=config.verify_ssl,
                allow_redirects=config.follow_redirects,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Parse response data
            try:
                response_data = response.json()
            except (json.JSONDecodeError, ValueError):
                response_data = response.text
            
            success = 200 <= response.status_code < 300
            
            if success:
                self._stats['successful_requests'] += 1
            else:
                self._stats['failed_requests'] += 1
            
            return RequestResult(
                success=success,
                status_code=response.status_code,
                data=response_data,
                headers=dict(response.headers),
                elapsed_ms=elapsed_ms,
                retries=retries,
            )
            
        except requests.exceptions.RetryError as e:
            self._stats['failed_requests'] += 1
            self._stats['total_retries'] += self.max_retries
            elapsed_ms = (time.time() - start_time) * 1000
            
            return RequestResult(
                success=False,
                status_code=None,
                data=None,
                headers={},
                elapsed_ms=elapsed_ms,
                retries=self.max_retries,
                error=f"Max retries exceeded: {e}",
            )
            
        except requests.exceptions.RequestException as e:
            self._stats['failed_requests'] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            
            return RequestResult(
                success=False,
                status_code=None,
                data=None,
                headers={},
                elapsed_ms=elapsed_ms,
                retries=retries,
                error=str(e),
            )
    
    def get(self, url: str, **kwargs) -> RequestResult:
        """Convenience method for GET requests."""
        return self.request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs) -> RequestResult:
        """Convenience method for POST requests."""
        return self.request("POST", url, **kwargs)
    
    def put(self, url: str, **kwargs) -> RequestResult:
        """Convenience method for PUT requests."""
        return self.request("PUT", url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> RequestResult:
        """Convenience method for DELETE requests."""
        return self.request("DELETE", url, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                **self._stats,
                'active_sessions': len(self._sessions),
            }
    
    def close(self):
        """Close all sessions in the pool."""
        with self._lock:
            for url, session in self._sessions.items():
                try:
                    session.close()
                except Exception:
                    pass
            self._sessions.clear()
            self._log.info("Closed all pooled connections")


# ==============================================================================
# RETRY QUEUE
# ==============================================================================

class RetryQueue:
    """
    Queue for failed requests that should be retried later.
    
    Implements priority-based ordering and automatic retry with backoff.
    """
    
    def __init__(
        self,
        connection_pool: Optional[ConnectionPool] = None,
        max_queue_size: int = 1000,
        retry_interval: float = 60.0,
    ):
        self.pool = connection_pool or ConnectionPool()
        self.max_queue_size = max_queue_size
        self.retry_interval = retry_interval
        
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._processing = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._log = get_logger("retry_queue")
        
        # Dead letter queue for permanently failed requests
        self._dead_letter: List[QueuedRequest] = []
        self._dead_letter_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'enqueued': 0,
            'processed': 0,
            'succeeded': 0,
            'permanently_failed': 0,
        }
    
    def enqueue(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        callback: Optional[Callable[[RequestResult], None]] = None,
    ) -> str:
        """
        Add a request to the retry queue.
        
        Returns:
            Request ID
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        request = QueuedRequest(
            id=request_id,
            method=method,
            url=url,
            headers=headers or {},
            data=data,
            priority=priority,
            created_at=time.time(),
            max_retries=max_retries,
            callback=callback,
        )
        
        try:
            self._queue.put_nowait(request)
            self._stats['enqueued'] += 1
            self._log.info(f"Enqueued request {request_id}: {method} {url}")
            return request_id
        except queue.Full:
            self._log.error("Retry queue is full, request dropped")
            raise RuntimeError("Retry queue is full")
    
    def _process_request(self, request: QueuedRequest) -> bool:
        """Process a single queued request."""
        self._log.info(f"Processing request {request.id} (retry {request.retry_count})")
        
        result = self.pool.request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            json_data=request.data if isinstance(request.data, (dict, list)) else None,
            data=request.data if isinstance(request.data, str) else None,
        )
        
        self._stats['processed'] += 1
        
        if result.success:
            self._stats['succeeded'] += 1
            self._log.info(f"Request {request.id} succeeded")
            
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    self._log.error(f"Callback error: {e}")
            
            return True
        else:
            request.retry_count += 1
            request.last_error = result.error
            
            if request.retry_count >= request.max_retries:
                # Move to dead letter queue
                with self._dead_letter_lock:
                    self._dead_letter.append(request)
                self._stats['permanently_failed'] += 1
                self._log.error(f"Request {request.id} permanently failed: {result.error}")
                return True  # Remove from queue
            else:
                # Re-enqueue with delay
                self._log.warn(f"Request {request.id} failed, will retry: {result.error}")
                return False
    
    def _worker(self):
        """Background worker for processing queue."""
        while not self._stop_event.is_set():
            try:
                request = self._queue.get(timeout=1.0)
                
                success = self._process_request(request)
                
                if not success:
                    # Wait before re-enqueueing
                    backoff = DEFAULT_BACKOFF_FACTOR ** request.retry_count
                    time.sleep(min(backoff, 60))
                    
                    try:
                        self._queue.put_nowait(request)
                    except queue.Full:
                        with self._dead_letter_lock:
                            self._dead_letter.append(request)
                
                self._queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self._log.error(f"Worker error: {e}")
    
    def start(self):
        """Start the background processing thread."""
        if self._processing:
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._processing = True
        self._log.info("Retry queue started")
    
    def stop(self, wait: bool = True, timeout: float = 10.0):
        """Stop the background processing thread."""
        if not self._processing:
            return
        
        self._stop_event.set()
        
        if wait and self._thread:
            self._thread.join(timeout=timeout)
        
        self._processing = False
        self._log.info("Retry queue stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'dead_letter_count': len(self._dead_letter),
            'processing': self._processing,
        }
    
    def get_dead_letter(self) -> List[Dict[str, Any]]:
        """Get failed requests from dead letter queue."""
        with self._dead_letter_lock:
            return [
                {
                    'id': r.id,
                    'method': r.method,
                    'url': r.url,
                    'retry_count': r.retry_count,
                    'created_at': datetime.fromtimestamp(r.created_at, timezone.utc).isoformat(),
                    'last_error': r.last_error,
                }
                for r in self._dead_letter
            ]
    
    def clear_dead_letter(self):
        """Clear the dead letter queue."""
        with self._dead_letter_lock:
            self._dead_letter.clear()


# ==============================================================================
# HTTP CLIENT WITH CIRCUIT BREAKER
# ==============================================================================

class ResilientHttpClient:
    """
    Production HTTP client combining connection pool, retry queue, and circuit breaker.
    
    This is the recommended client for all HTTP operations in the agent.
    """
    
    _instance: Optional["ResilientHttpClient"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        enable_circuit_breaker: bool = True,
        enable_retry_queue: bool = True,
    ):
        self.pool = ConnectionPool(pool_size=pool_size)
        self.retry_queue = RetryQueue(self.pool) if enable_retry_queue else None
        self._enable_circuit_breaker = enable_circuit_breaker
        self._circuit_breakers: Dict[str, Any] = {}
        self._log = get_logger("http_client")
        
        # Start retry queue if enabled
        if self.retry_queue:
            self.retry_queue.start()
    
    @classmethod
    def get_instance(cls, **kwargs) -> "ResilientHttpClient":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance
    
    def _get_circuit_breaker(self, base_url: str):
        """Get or create circuit breaker for a base URL."""
        if not self._enable_circuit_breaker:
            return None
        
        if base_url not in self._circuit_breakers:
            try:
                from agent.circuit_breaker import CircuitBreaker
                self._circuit_breakers[base_url] = CircuitBreaker(
                    name=base_url,
                    failure_threshold=5,
                    reset_timeout=60.0,
                )
            except ImportError:
                return None
        
        return self._circuit_breakers.get(base_url)
    
    async def request_async(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        json_data: Any = None,
        config: Optional[RequestConfig] = None,
        queue_on_failure: bool = False,
    ) -> RequestResult:
        """
        Make an async HTTP request with circuit breaker protection.
        
        Args:
            method: HTTP method
            url: Full URL
            headers: Request headers
            data: Form data
            json_data: JSON data
            config: Request configuration
            queue_on_failure: Add to retry queue on failure
        
        Returns:
            RequestResult
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        circuit_breaker = self._get_circuit_breaker(base_url)
        
        # Wrap request in circuit breaker if available
        if circuit_breaker:
            try:
                from agent.circuit_breaker import CircuitBreakerError
                
                async def _make_request():
                    return self.pool.request(
                        method=method,
                        url=url,
                        headers=headers,
                        data=data,
                        json_data=json_data,
                        config=config,
                    )
                
                result = await circuit_breaker.call(_make_request)
                
            except CircuitBreakerError as e:
                return RequestResult(
                    success=False,
                    status_code=None,
                    data=None,
                    headers={},
                    elapsed_ms=0,
                    retries=0,
                    error=str(e),
                )
        else:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pool.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json_data=json_data,
                    config=config,
                )
            )
        
        # Queue for retry if failed and enabled
        if not result.success and queue_on_failure and self.retry_queue:
            self.retry_queue.enqueue(
                method=method,
                url=url,
                headers=headers,
                data=json_data or data,
            )
        
        return result
    
    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        json_data: Any = None,
        config: Optional[RequestConfig] = None,
        queue_on_failure: bool = False,
    ) -> RequestResult:
        """
        Make a synchronous HTTP request.
        """
        result = self.pool.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            json_data=json_data,
            config=config,
        )
        
        if not result.success and queue_on_failure and self.retry_queue:
            self.retry_queue.enqueue(
                method=method,
                url=url,
                headers=headers,
                data=json_data or data,
            )
        
        return result
    
    def get(self, url: str, **kwargs) -> RequestResult:
        """GET request."""
        return self.request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs) -> RequestResult:
        """POST request."""
        return self.request("POST", url, **kwargs)
    
    def put(self, url: str, **kwargs) -> RequestResult:
        """PUT request."""
        return self.request("PUT", url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> RequestResult:
        """DELETE request."""
        return self.request("DELETE", url, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {
            'connection_pool': self.pool.get_stats(),
        }
        
        if self.retry_queue:
            stats['retry_queue'] = self.retry_queue.get_stats()
        
        if self._circuit_breakers:
            stats['circuit_breakers'] = {
                url: {
                    'state': cb.state.value,
                    'stats': {
                        'total_calls': cb.stats.total_calls,
                        'failed_calls': cb.stats.failed_calls,
                    }
                }
                for url, cb in self._circuit_breakers.items()
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown the client."""
        if self.retry_queue:
            self.retry_queue.stop()
        self.pool.close()
        self._log.info("HTTP client shutdown complete")


# ==============================================================================
# SINGLETON ACCESS
# ==============================================================================

def get_http_client(**kwargs) -> ResilientHttpClient:
    """Get the global HTTP client instance."""
    return ResilientHttpClient.get_instance(**kwargs)


def get_connection_pool() -> ConnectionPool:
    """Get a connection pool."""
    return get_http_client().pool
