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
]

