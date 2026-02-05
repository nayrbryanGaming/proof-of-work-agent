"""
Comprehensive Error Codes and Exception Hierarchy.
Provides structured error handling for the entire agent system.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Type, Union


# ============================================================
# Error Categories
# ============================================================

class ErrorCategory(str, Enum):
    """High-level error categories."""
    SYSTEM = "SYSTEM"           # Internal system errors
    NETWORK = "NETWORK"         # Network/connectivity errors
    API = "API"                 # External API errors
    AUTH = "AUTH"               # Authentication/authorization
    VALIDATION = "VALIDATION"   # Input validation
    STATE = "STATE"             # State management
    CRYPTO = "CRYPTO"           # Cryptographic operations
    SOLANA = "SOLANA"           # Solana blockchain
    TASK = "TASK"               # Task processing
    CONFIG = "CONFIG"           # Configuration
    RESOURCE = "RESOURCE"       # Resource limits


class ErrorSeverity(IntEnum):
    """Error severity levels."""
    DEBUG = 10       # Debug information
    INFO = 20        # Informational
    WARNING = 30     # Warning, can continue
    ERROR = 40       # Error, operation failed
    CRITICAL = 50    # Critical, system may be unstable
    FATAL = 60       # Fatal, system must stop


# ============================================================
# Error Codes
# ============================================================

class ErrorCode(str, Enum):
    """
    Enumerated error codes with format: CATEGORY_SPECIFIC_ERROR
    
    Format: XX_YYY_ZZZ
    - XX: Category (SY=System, NE=Network, AP=API, etc.)
    - YYY: Subcategory
    - ZZZ: Specific error
    """
    
    # ============ SYSTEM ERRORS (SY) ============
    SY_INIT_FAILED = "SY_001_001"
    SY_SHUTDOWN_FAILED = "SY_001_002"
    SY_MODULE_LOAD_FAILED = "SY_002_001"
    SY_DEPENDENCY_MISSING = "SY_002_002"
    SY_MEMORY_EXHAUSTED = "SY_003_001"
    SY_DISK_FULL = "SY_003_002"
    SY_THREAD_POOL_EXHAUSTED = "SY_004_001"
    SY_ASYNC_TASK_FAILED = "SY_004_002"
    SY_SIGNAL_HANDLER_FAILED = "SY_005_001"
    SY_UNEXPECTED_ERROR = "SY_999_999"
    
    # ============ NETWORK ERRORS (NE) ============
    NE_CONNECTION_REFUSED = "NE_001_001"
    NE_CONNECTION_TIMEOUT = "NE_001_002"
    NE_CONNECTION_RESET = "NE_001_003"
    NE_DNS_RESOLUTION_FAILED = "NE_002_001"
    NE_SSL_CERTIFICATE_ERROR = "NE_003_001"
    NE_SSL_HANDSHAKE_FAILED = "NE_003_002"
    NE_PROXY_ERROR = "NE_004_001"
    NE_HOST_UNREACHABLE = "NE_005_001"
    NE_NETWORK_UNREACHABLE = "NE_005_002"
    
    # ============ API ERRORS (AP) ============
    AP_REQUEST_FAILED = "AP_001_001"
    AP_RESPONSE_INVALID = "AP_001_002"
    AP_RESPONSE_TIMEOUT = "AP_001_003"
    AP_RATE_LIMITED = "AP_002_001"
    AP_QUOTA_EXCEEDED = "AP_002_002"
    AP_SERVER_ERROR = "AP_003_001"
    AP_SERVICE_UNAVAILABLE = "AP_003_002"
    AP_MAINTENANCE_MODE = "AP_003_003"
    AP_NOT_FOUND = "AP_004_001"
    AP_METHOD_NOT_ALLOWED = "AP_004_002"
    AP_CONFLICT = "AP_004_003"
    AP_OPENAI_ERROR = "AP_010_001"
    AP_OPENAI_RATE_LIMITED = "AP_010_002"
    AP_OPENAI_QUOTA_EXCEEDED = "AP_010_003"
    AP_COLOSSEUM_ERROR = "AP_020_001"
    AP_COLOSSEUM_RATE_LIMITED = "AP_020_002"
    
    # ============ AUTHENTICATION ERRORS (AU) ============
    AU_INVALID_CREDENTIALS = "AU_001_001"
    AU_TOKEN_EXPIRED = "AU_001_002"
    AU_TOKEN_INVALID = "AU_001_003"
    AU_PERMISSION_DENIED = "AU_002_001"
    AU_INSUFFICIENT_SCOPE = "AU_002_002"
    AU_SESSION_EXPIRED = "AU_003_001"
    AU_WALLET_NOT_CONNECTED = "AU_004_001"
    AU_SIGNATURE_INVALID = "AU_004_002"
    
    # ============ VALIDATION ERRORS (VA) ============
    VA_REQUIRED_FIELD_MISSING = "VA_001_001"
    VA_FIELD_TOO_SHORT = "VA_001_002"
    VA_FIELD_TOO_LONG = "VA_001_003"
    VA_INVALID_FORMAT = "VA_002_001"
    VA_INVALID_TYPE = "VA_002_002"
    VA_INVALID_RANGE = "VA_002_003"
    VA_INVALID_EMAIL = "VA_003_001"
    VA_INVALID_URL = "VA_003_002"
    VA_INVALID_JSON = "VA_003_003"
    VA_SCHEMA_MISMATCH = "VA_004_001"
    VA_CONSTRAINT_VIOLATION = "VA_004_002"
    
    # ============ STATE ERRORS (ST) ============
    ST_FILE_NOT_FOUND = "ST_001_001"
    ST_FILE_CORRUPTED = "ST_001_002"
    ST_FILE_LOCKED = "ST_001_003"
    ST_STATE_INVALID = "ST_002_001"
    ST_STATE_OUTDATED = "ST_002_002"
    ST_CHECKSUM_MISMATCH = "ST_002_003"
    ST_RECOVERY_FAILED = "ST_003_001"
    ST_BACKUP_FAILED = "ST_003_002"
    ST_PERSISTENCE_FAILED = "ST_004_001"
    
    # ============ CRYPTO ERRORS (CR) ============
    CR_KEY_GENERATION_FAILED = "CR_001_001"
    CR_KEY_LOAD_FAILED = "CR_001_002"
    CR_KEY_INVALID = "CR_001_003"
    CR_SIGNING_FAILED = "CR_002_001"
    CR_SIGNATURE_INVALID = "CR_002_002"
    CR_VERIFICATION_FAILED = "CR_002_003"
    CR_ENCRYPTION_FAILED = "CR_003_001"
    CR_DECRYPTION_FAILED = "CR_003_002"
    CR_HASH_FAILED = "CR_004_001"
    
    # ============ SOLANA ERRORS (SO) ============
    SO_CONNECTION_FAILED = "SO_001_001"
    SO_RPC_ERROR = "SO_001_002"
    SO_RPC_TIMEOUT = "SO_001_003"
    SO_TRANSACTION_FAILED = "SO_002_001"
    SO_TRANSACTION_TIMEOUT = "SO_002_002"
    SO_TRANSACTION_REJECTED = "SO_002_003"
    SO_INSUFFICIENT_FUNDS = "SO_003_001"
    SO_ACCOUNT_NOT_FOUND = "SO_003_002"
    SO_PROGRAM_ERROR = "SO_004_001"
    SO_INSTRUCTION_ERROR = "SO_004_002"
    SO_BLOCKHASH_EXPIRED = "SO_005_001"
    SO_AIRDROP_FAILED = "SO_006_001"
    SO_AIRDROP_RATE_LIMITED = "SO_006_002"
    
    # ============ TASK ERRORS (TA) ============
    TA_TASK_NOT_FOUND = "TA_001_001"
    TA_TASK_INVALID = "TA_001_002"
    TA_TASK_EXPIRED = "TA_001_003"
    TA_PROCESSING_FAILED = "TA_002_001"
    TA_TIMEOUT = "TA_002_002"
    TA_DUPLICATE = "TA_003_001"
    TA_QUEUE_FULL = "TA_004_001"
    TA_QUEUE_EMPTY = "TA_004_002"
    
    # ============ CONFIG ERRORS (CF) ============
    CF_MISSING_REQUIRED = "CF_001_001"
    CF_INVALID_VALUE = "CF_001_002"
    CF_FILE_NOT_FOUND = "CF_002_001"
    CF_PARSE_ERROR = "CF_002_002"
    CF_ENV_VAR_MISSING = "CF_003_001"
    
    # ============ RESOURCE ERRORS (RE) ============
    RE_CPU_LIMIT = "RE_001_001"
    RE_MEMORY_LIMIT = "RE_001_002"
    RE_DISK_LIMIT = "RE_001_003"
    RE_RATE_LIMIT = "RE_002_001"
    RE_QUOTA_EXCEEDED = "RE_002_002"
    RE_CIRCUIT_OPEN = "RE_003_001"
    
    @property
    def category(self) -> str:
        """Get error category from code."""
        prefix = self.value.split("_")[0]
        mapping = {
            "SY": ErrorCategory.SYSTEM,
            "NE": ErrorCategory.NETWORK,
            "AP": ErrorCategory.API,
            "AU": ErrorCategory.AUTH,
            "VA": ErrorCategory.VALIDATION,
            "ST": ErrorCategory.STATE,
            "CR": ErrorCategory.CRYPTO,
            "SO": ErrorCategory.SOLANA,
            "TA": ErrorCategory.TASK,
            "CF": ErrorCategory.CONFIG,
            "RE": ErrorCategory.RESOURCE,
        }
        return mapping.get(prefix, ErrorCategory.SYSTEM).value


# ============================================================
# Error Details
# ============================================================

@dataclass
class ErrorContext:
    """Context information for an error."""
    
    code: ErrorCode
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trace_id: Optional[str] = None
    module: Optional[str] = None
    operation: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    cause: Optional["ErrorContext"] = None
    retry_after: Optional[float] = None
    recoverable: bool = True
    
    def __post_init__(self):
        if self.category is None:
            self.category = self.code.category
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.name,
            "category": self.category,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
        }
        
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.module:
            result["module"] = self.module
        if self.operation:
            result["operation"] = self.operation
        if self.details:
            result["details"] = self.details
        if self.stack_trace:
            result["stack_trace"] = self.stack_trace
        if self.retry_after:
            result["retry_after"] = self.retry_after
        if self.cause:
            result["cause"] = self.cause.to_dict()
        
        return result
    
    def to_log_message(self) -> str:
        """Format for logging."""
        parts = [f"[{self.code.value}]", self.message]
        
        if self.module:
            parts.insert(0, f"[{self.module}]")
        if self.operation:
            parts.append(f"(operation: {self.operation})")
        
        return " ".join(parts)


# ============================================================
# Base Exceptions
# ============================================================

class AgentError(Exception):
    """Base exception for all agent errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SY_UNEXPECTED_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Capture stack trace if cause is provided
        if cause:
            self.stack_trace = "".join(traceback.format_exception(
                type(cause), cause, cause.__traceback__
            ))
        else:
            self.stack_trace = None
    
    @property
    def context(self) -> ErrorContext:
        """Get error context."""
        cause_context = None
        if self.cause and isinstance(self.cause, AgentError):
            cause_context = self.cause.context
        
        return ErrorContext(
            code=self.code,
            message=self.message,
            severity=self.severity,
            details=self.details,
            stack_trace=self.stack_trace,
            cause=cause_context,
            retry_after=self.retry_after,
            recoverable=self.recoverable,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.context.to_dict()
    
    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"


# ============================================================
# Specific Exception Classes
# ============================================================

class SystemError(AgentError):
    """System-level errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.SY_UNEXPECTED_ERROR, **kwargs):
        super().__init__(message, code, **kwargs)


class NetworkError(AgentError):
    """Network-related errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.NE_CONNECTION_REFUSED, **kwargs):
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("retry_after", 5.0)
        super().__init__(message, code, **kwargs)


class APIError(AgentError):
    """API-related errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.AP_REQUEST_FAILED,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate
        
        super().__init__(message, code, details=details, **kwargs)
        self.status_code = status_code


class RateLimitError(APIError):
    """Rate limit errors."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float = 60.0,
        **kwargs
    ):
        super().__init__(
            message,
            code=ErrorCode.AP_RATE_LIMITED,
            retry_after=retry_after,
            **kwargs
        )


class AuthenticationError(AgentError):
    """Authentication/authorization errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.AU_INVALID_CREDENTIALS, **kwargs):
        kwargs.setdefault("recoverable", False)
        super().__init__(message, code, **kwargs)


class ValidationError(AgentError):
    """Validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        code: ErrorCode = ErrorCode.VA_REQUIRED_FIELD_MISSING,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        
        kwargs.setdefault("recoverable", False)
        super().__init__(message, code, details=details, **kwargs)


class StateError(AgentError):
    """State management errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.ST_STATE_INVALID, **kwargs):
        super().__init__(message, code, **kwargs)


class CryptoError(AgentError):
    """Cryptographic operation errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.CR_KEY_INVALID, **kwargs):
        kwargs.setdefault("recoverable", False)
        super().__init__(message, code, **kwargs)


class SolanaError(AgentError):
    """Solana blockchain errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SO_RPC_ERROR,
        tx_signature: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if tx_signature:
            details["tx_signature"] = tx_signature
        
        super().__init__(message, code, details=details, **kwargs)


class TaskError(AgentError):
    """Task processing errors."""
    
    def __init__(
        self,
        message: str,
        task_id: Optional[int] = None,
        code: ErrorCode = ErrorCode.TA_PROCESSING_FAILED,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if task_id:
            details["task_id"] = task_id
        
        super().__init__(message, code, details=details, **kwargs)


class ConfigurationError(AgentError):
    """Configuration errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.CF_MISSING_REQUIRED, **kwargs):
        kwargs.setdefault("recoverable", False)
        super().__init__(message, code, **kwargs)


class ResourceError(AgentError):
    """Resource limit errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.RE_RATE_LIMIT, **kwargs):
        super().__init__(message, code, **kwargs)


class CircuitBreakerError(ResourceError):
    """Circuit breaker is open."""
    
    def __init__(
        self,
        service_name: str,
        retry_after: float = 60.0,
        **kwargs
    ):
        message = f"Circuit breaker open for service: {service_name}"
        details = kwargs.pop("details", {})
        details["service_name"] = service_name
        
        super().__init__(
            message,
            code=ErrorCode.RE_CIRCUIT_OPEN,
            retry_after=retry_after,
            details=details,
            **kwargs
        )


# ============================================================
# Error Registry
# ============================================================

class ErrorRegistry:
    """Registry for error tracking and statistics."""
    
    def __init__(self):
        self.errors: List[ErrorContext] = []
        self.counts: Dict[str, int] = {}
        self.max_errors = 1000
    
    def record(self, error: Union[AgentError, ErrorContext]) -> None:
        """Record an error."""
        if isinstance(error, AgentError):
            context = error.context
        else:
            context = error
        
        self.errors.append(context)
        
        # Update counts
        code = context.code.value
        self.counts[code] = self.counts.get(code, 0) + 1
        
        # Prune old errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
    
    def get_recent(self, count: int = 10) -> List[ErrorContext]:
        """Get recent errors."""
        return self.errors[-count:]
    
    def get_by_code(self, code: ErrorCode) -> List[ErrorContext]:
        """Get errors by code."""
        return [e for e in self.errors if e.code == code]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total": len(self.errors),
            "by_code": dict(self.counts),
            "by_category": self._count_by_category(),
            "by_severity": self._count_by_severity(),
        }
    
    def _count_by_category(self) -> Dict[str, int]:
        counts = {}
        for error in self.errors:
            cat = error.category or "unknown"
            counts[cat] = counts.get(cat, 0) + 1
        return counts
    
    def _count_by_severity(self) -> Dict[str, int]:
        counts = {}
        for error in self.errors:
            sev = error.severity.name
            counts[sev] = counts.get(sev, 0) + 1
        return counts
    
    def clear(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()
        self.counts.clear()


# Global registry
_error_registry: Optional[ErrorRegistry] = None


def get_error_registry() -> ErrorRegistry:
    """Get global error registry."""
    global _error_registry
    if _error_registry is None:
        _error_registry = ErrorRegistry()
    return _error_registry


def record_error(error: Union[AgentError, Exception]) -> None:
    """Record an error in the global registry."""
    if isinstance(error, AgentError):
        get_error_registry().record(error)
    else:
        # Wrap generic exception
        wrapped = AgentError(str(error), cause=error)
        get_error_registry().record(wrapped)


# ============================================================
# Error Handling Utilities
# ============================================================

def wrap_exception(
    exc: Exception,
    code: ErrorCode = ErrorCode.SY_UNEXPECTED_ERROR,
    message: Optional[str] = None,
    **kwargs
) -> AgentError:
    """Wrap a generic exception in an AgentError."""
    msg = message or str(exc)
    return AgentError(msg, code=code, cause=exc, **kwargs)


def is_retryable(error: Union[AgentError, Exception]) -> bool:
    """Check if an error is retryable."""
    if isinstance(error, AgentError):
        return error.recoverable and error.retry_after is not None
    
    # Check for common retryable exception types
    error_str = str(type(error).__name__).lower()
    retryable_patterns = ["timeout", "connection", "temporary", "ratelimit"]
    return any(p in error_str for p in retryable_patterns)


def get_retry_delay(error: Union[AgentError, Exception], default: float = 5.0) -> float:
    """Get retry delay from error."""
    if isinstance(error, AgentError) and error.retry_after:
        return error.retry_after
    return default
