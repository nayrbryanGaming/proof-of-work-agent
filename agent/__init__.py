"""
Proof-of-Work Agent - Core Agent Module

This module provides the core functionality for the autonomous AI agent:
- Configuration management
- Structured logging
- Heartbeat monitoring
- AI decision engine
- State management
- Input validation
- Circuit breaker fault tolerance
- Advanced metrics collection
- Event-driven architecture
- Retry strategies
- Health monitoring
- Task queue system
- Job scheduling
- Rate limiting
"""

from .config import config
from .logger import get_logger
from .heartbeat import HeartbeatChecker
from .decision import DecisionEngine
from .state import StateManager, get_state_manager
from .validators import (
    ValidationError,
    ValidationResult,
    validate_config,
    validate_task,
    validate_forum_comment,
    validate_solana_address,
    compute_hash,
    verify_hash,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
    registry as circuit_registry,
)

# Advanced modules
from .metrics import (
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
    registry as metrics_registry,
    agent_metrics,
)
from .events import (
    Event,
    EventType,
    EventEmitter,
    EventPriority,
    event_bus,
    emit,
    emit_async,
    on,
    off,
)
from .retry import (
    Retrier,
    RetryConfig,
    RetryStrategy,
    RetryResult,
    retry,
    AdaptiveRetrier,
    RetryBudget,
)
from .health import (
    HealthStatus,
    HealthCheck,
    HealthCheckResult,
    HealthCheckRegistry,
    HealthMonitor,
    health_registry,
    register_health_check,
)
from .task_queue import (
    TaskQueue,
    TaskDefinition,
    TaskStatus,
    TaskPriority,
    TaskHandler,
    task_queue,
    task,
)
from .scheduler import (
    Scheduler,
    ScheduledJob,
    ScheduleType,
    CronExpression,
    scheduler,
    interval,
    cron,
    daily,
)
from .rate_limiter import (
    RateLimiter,
    TokenBucket,
    SlidingWindowLog,
    FixedWindowCounter,
    LeakyBucket,
    RateLimitExceeded,
    rate_limit,
    colosseum_limiter,
    openai_limiter,
    solana_limiter,
)

# Self-healing and shutdown
from .watchdog import Watchdog, get_watchdog, RecoveryAction
from .shutdown import GracefulShutdown, get_shutdown_handler

# New advanced modules (v2.1)
from .crypto import (
    Ed25519,
    Base58,
    SolanaKeypair,
    ProofSigner,
    SignedProof,
    ProofHashchain,
    get_signer,
    sign_result,
    get_wallet_address,
)
from .backup import (
    BackupManager,
    RecoveryManager,
    StateSnapshot,
    BackupManifest,
    BackupType,
    get_backup_manager,
    get_state_snapshot,
    quick_backup,
    quick_snapshot,
)
from .errors import (
    AgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
    ErrorRegistry,
    NetworkError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ValidationError as AgentValidationError,
    StateError,
    CryptoError,
    SolanaError,
    TaskError,
    ConfigurationError,
    ResourceError,
    CircuitBreakerError as CBError,
    get_error_registry,
    record_error,
    wrap_exception,
    is_retryable,
    get_retry_delay,
)
from .telemetry import (
    TelemetryManager,
    TraceContext,
    Span,
    SpanKind,
    SpanStatus,
    Tracer,
    Profiler,
    SystemMonitor,
    get_telemetry,
    traced,
    profiled,
)

__all__ = [
    # Config
    "config",
    # Logging
    "get_logger",
    # Core components
    "HeartbeatChecker",
    "DecisionEngine",
    # State management
    "StateManager",
    "get_state_manager",
    # Validation
    "ValidationError",
    "ValidationResult",
    "validate_config",
    "validate_task",
    "validate_forum_comment",
    "validate_solana_address",
    "compute_hash",
    "verify_hash",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "circuit_breaker",
    "circuit_registry",
    # Crypto (v2.1)
    "Ed25519",
    "Base58",
    "SolanaKeypair",
    "ProofSigner",
    "SignedProof",
    "ProofHashchain",
    "get_signer",
    "sign_result",
    "get_wallet_address",
    # Backup (v2.1)
    "BackupManager",
    "RecoveryManager",
    "StateSnapshot",
    "get_backup_manager",
    "quick_backup",
    "quick_snapshot",
    # Errors (v2.1)
    "AgentError",
    "ErrorCode",
    "ErrorCategory",
    "get_error_registry",
    "record_error",
    # Telemetry (v2.1)
    "TelemetryManager",
    "get_telemetry",
    "traced",
    "profiled",
    # Watchdog
    "Watchdog",
    "get_watchdog",
    # Shutdown
    "GracefulShutdown",
    "get_shutdown_handler",
]

