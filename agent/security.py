"""
Security Module for Proof-of-Work Agent
========================================

Comprehensive security hardening including:
- DDoS protection with rate limiting
- Input sanitization and validation
- Secret management
- Request fingerprinting
- IP blocking (for API mode)
- Encryption utilities
- HMAC verification
- Nonce generation for replay attack prevention

This module is CRITICAL for production deployment.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agent.logger import get_logger


# ==============================================================================
# SECURITY CONSTANTS
# ==============================================================================

# Maximum request size (10 KB default)
MAX_REQUEST_SIZE = 10 * 1024

# Maximum string length for user inputs
MAX_INPUT_LENGTH = 10000

# Nonce expiration (5 minutes)
NONCE_EXPIRATION_SECONDS = 300

# Rate limit window
RATE_LIMIT_WINDOW = 60  # seconds

# Maximum requests per window
MAX_REQUESTS_PER_WINDOW = 100

# Suspicious patterns that should be rejected
SUSPICIOUS_PATTERNS = [
    r'<script\b',
    r'javascript:',
    r'on\w+\s*=',
    r'eval\s*\(',
    r'exec\s*\(',
    r'__import__',
    r'\bos\.system\b',
    r'\bsubprocess\b',
    r'\.\./\.\./',  # Path traversal
    r'%00',  # Null byte
]

# Allowed characters for identifiers
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

# Base32 alphabet for secure token generation
TOKEN_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SecurityEvent:
    """Record of a security-related event."""
    event_type: str  # 'rate_limit', 'invalid_input', 'blocked', 'suspicious'
    timestamp: datetime
    source: str
    details: Dict[str, Any]
    severity: str = 'warning'  # 'info', 'warning', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'details': self.details,
            'severity': self.severity
        }


@dataclass
class RateLimitEntry:
    """Tracks rate limiting for a single source."""
    source: str
    window_start: float
    request_count: int = 0
    blocked_until: Optional[float] = None


@dataclass
class SecurityConfig:
    """Security configuration."""
    max_request_size: int = MAX_REQUEST_SIZE
    max_input_length: int = MAX_INPUT_LENGTH
    rate_limit_window: int = RATE_LIMIT_WINDOW
    max_requests_per_window: int = MAX_REQUESTS_PER_WINDOW
    nonce_expiration: int = NONCE_EXPIRATION_SECONDS
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    block_suspicious_patterns: bool = True
    log_security_events: bool = True


# ==============================================================================
# RATE LIMITER
# ==============================================================================

class SecurityRateLimiter:
    """
    Thread-safe rate limiter with sliding window.
    Prevents DDoS and brute force attacks.
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._entries: Dict[str, RateLimitEntry] = {}
        self._lock = threading.Lock()
        self._blocked_sources: Set[str] = set()
        self._permanent_blocks: Set[str] = set()
        self._log = get_logger("security.rate_limiter")
    
    def check(self, source: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a request from source is allowed.
        
        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        with self._lock:
            now = time.time()
            
            # Check permanent blocks
            if source in self._permanent_blocks:
                return False, "Permanently blocked"
            
            # Get or create entry
            entry = self._entries.get(source)
            
            if entry is None:
                entry = RateLimitEntry(
                    source=source,
                    window_start=now,
                    request_count=1
                )
                self._entries[source] = entry
                return True, None
            
            # Check if blocked
            if entry.blocked_until and now < entry.blocked_until:
                remaining = int(entry.blocked_until - now)
                return False, f"Blocked for {remaining} more seconds"
            
            # Check if window expired
            window_elapsed = now - entry.window_start
            if window_elapsed >= self.config.rate_limit_window:
                # Reset window
                entry.window_start = now
                entry.request_count = 1
                entry.blocked_until = None
                return True, None
            
            # Increment counter
            entry.request_count += 1
            
            # Check limit
            if entry.request_count > self.config.max_requests_per_window:
                # Calculate block duration (exponential backoff)
                over_limit = entry.request_count - self.config.max_requests_per_window
                block_duration = min(60 * 2 ** min(over_limit // 10, 5), 3600)  # Max 1 hour
                entry.blocked_until = now + block_duration
                
                self._log.warn(
                    f"Rate limit exceeded for {source}: "
                    f"{entry.request_count} requests, blocked for {block_duration}s"
                )
                
                return False, f"Rate limit exceeded, blocked for {block_duration}s"
            
            return True, None
    
    def block_source(self, source: str, duration: int = 3600):
        """Temporarily block a source."""
        with self._lock:
            entry = self._entries.get(source)
            if entry is None:
                entry = RateLimitEntry(
                    source=source,
                    window_start=time.time(),
                    request_count=0
                )
                self._entries[source] = entry
            
            entry.blocked_until = time.time() + duration
            self._blocked_sources.add(source)
            self._log.warn(f"Blocked source {source} for {duration}s")
    
    def permanent_block(self, source: str):
        """Permanently block a source."""
        with self._lock:
            self._permanent_blocks.add(source)
            self._log.error(f"Permanently blocked source: {source}")
    
    def unblock(self, source: str):
        """Unblock a source."""
        with self._lock:
            if source in self._blocked_sources:
                self._blocked_sources.discard(source)
            if source in self._permanent_blocks:
                self._permanent_blocks.discard(source)
            if source in self._entries:
                self._entries[source].blocked_until = None
            self._log.info(f"Unblocked source: {source}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                'active_entries': len(self._entries),
                'blocked_sources': len(self._blocked_sources),
                'permanent_blocks': len(self._permanent_blocks),
                'total_requests': sum(e.request_count for e in self._entries.values())
            }
    
    def cleanup(self):
        """Remove expired entries."""
        with self._lock:
            now = time.time()
            expired = [
                source for source, entry in self._entries.items()
                if now - entry.window_start > self.config.rate_limit_window * 2
                and entry.blocked_until is None
            ]
            for source in expired:
                del self._entries[source]
            return len(expired)


# ==============================================================================
# INPUT VALIDATOR
# ==============================================================================

class InputValidator:
    """
    Validates and sanitizes all inputs.
    Prevents injection attacks and malformed data.
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._suspicious_patterns = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]
        self._log = get_logger("security.validator")
    
    def validate_string(self, value: str, max_length: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a string input.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, str):
            return False, "Input must be a string"
        
        max_len = max_length or self.config.max_input_length
        if len(value) > max_len:
            return False, f"Input exceeds maximum length of {max_len}"
        
        if self.config.block_suspicious_patterns:
            for pattern in self._suspicious_patterns:
                if pattern.search(value):
                    self._log.warn(f"Suspicious pattern detected: {pattern.pattern}")
                    return False, "Input contains suspicious content"
        
        return True, None
    
    def sanitize_string(self, value: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize a string by removing dangerous content.
        """
        if not isinstance(value, str):
            return ""
        
        # Truncate to max length
        max_len = max_length or self.config.max_input_length
        result = value[:max_len]
        
        # Remove null bytes
        result = result.replace('\x00', '')
        
        # Remove suspicious patterns
        if self.config.block_suspicious_patterns:
            for pattern in self._suspicious_patterns:
                result = pattern.sub('', result)
        
        return result.strip()
    
    def validate_identifier(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate an identifier (alphanumeric, underscore, hyphen)."""
        if not value:
            return False, "Identifier cannot be empty"
        
        if len(value) > 256:
            return False, "Identifier too long"
        
        if not SAFE_ID_PATTERN.match(value):
            return False, "Identifier contains invalid characters"
        
        return True, None
    
    def validate_json_size(self, data: bytes) -> Tuple[bool, Optional[str]]:
        """Validate JSON payload size."""
        if len(data) > self.config.max_request_size:
            return False, f"Payload exceeds maximum size of {self.config.max_request_size} bytes"
        return True, None
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """Validate URL to prevent SSRF attacks."""
        if not url:
            return False, "URL cannot be empty"
        
        # Check for path traversal
        if '..' in url or '%2e%2e' in url.lower():
            return False, "Path traversal detected"
        
        # Check for local addresses (SSRF prevention)
        dangerous_hosts = [
            '127.0.0.1', 'localhost', '0.0.0.0',
            '169.254', '10.', '172.16', '172.17', '172.18',
            '172.19', '172.20', '172.21', '172.22', '172.23',
            '172.24', '172.25', '172.26', '172.27', '172.28',
            '172.29', '172.30', '172.31', '192.168'
        ]
        
        url_lower = url.lower()
        for host in dangerous_hosts:
            if host in url_lower:
                return False, "Internal addresses not allowed"
        
        return True, None


# ==============================================================================
# NONCE MANAGER
# ==============================================================================

class NonceManager:
    """
    Manages nonces for replay attack prevention.
    Each nonce can only be used once within its expiration window.
    """
    
    def __init__(self, expiration_seconds: int = NONCE_EXPIRATION_SECONDS):
        self._expiration = expiration_seconds
        self._used_nonces: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._log = get_logger("security.nonce")
    
    def generate(self) -> str:
        """Generate a new secure nonce."""
        # 32 bytes = 256 bits of entropy
        nonce_bytes = secrets.token_bytes(32)
        nonce = base64.urlsafe_b64encode(nonce_bytes).decode('ascii').rstrip('=')
        return nonce
    
    def validate_and_consume(self, nonce: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a nonce and mark it as used.
        Returns (is_valid, error_message)
        """
        with self._lock:
            now = time.time()
            
            # Cleanup expired nonces
            expired = [n for n, t in self._used_nonces.items() if now - t > self._expiration]
            for n in expired:
                del self._used_nonces[n]
            
            # Check if already used
            if nonce in self._used_nonces:
                self._log.warn(f"Nonce replay attempt detected")
                return False, "Nonce already used"
            
            # Mark as used
            self._used_nonces[nonce] = now
            return True, None
    
    def generate_timestamped(self) -> Tuple[str, float]:
        """Generate a nonce with timestamp."""
        nonce = self.generate()
        timestamp = time.time()
        return nonce, timestamp


# ==============================================================================
# HMAC VERIFIER
# ==============================================================================

class HMACVerifier:
    """
    HMAC-based message authentication.
    Used to verify request integrity and authenticity.
    """
    
    def __init__(self, secret_key: Optional[bytes] = None):
        if secret_key:
            self._key = secret_key
        else:
            # Generate a random key if not provided
            self._key = secrets.token_bytes(32)
        self._log = get_logger("security.hmac")
    
    @classmethod
    def from_env(cls, env_var: str = "HMAC_SECRET") -> "HMACVerifier":
        """Create from environment variable."""
        secret = os.getenv(env_var, "")
        if secret:
            key = hashlib.sha256(secret.encode()).digest()
        else:
            key = None
        return cls(key)
    
    def sign(self, message: bytes) -> str:
        """Generate HMAC signature for a message."""
        sig = hmac.new(self._key, message, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(sig).decode('ascii')
    
    def verify(self, message: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        try:
            expected_sig = base64.urlsafe_b64decode(signature)
            actual_sig = hmac.new(self._key, message, hashlib.sha256).digest()
            return hmac.compare_digest(expected_sig, actual_sig)
        except Exception:
            return False
    
    def sign_dict(self, data: Dict[str, Any]) -> str:
        """Sign a dictionary (JSON-serialized)."""
        import json
        message = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        return self.sign(message)
    
    def verify_dict(self, data: Dict[str, Any], signature: str) -> bool:
        """Verify a signed dictionary."""
        import json
        message = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        return self.verify(message, signature)


# ==============================================================================
# SECRET MANAGER
# ==============================================================================

class SecretManager:
    """
    Secure secret management with encryption.
    Protects sensitive configuration values.
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self._key = master_key
        else:
            # Derive key from environment
            secret = os.getenv("MASTER_KEY", os.getenv("SECRET_KEY", ""))
            if secret:
                self._key = hashlib.pbkdf2_hmac(
                    'sha256',
                    secret.encode(),
                    b'pow-agent-salt-v1',
                    100000
                )
            else:
                # Generate ephemeral key (secrets won't persist across restarts)
                self._key = secrets.token_bytes(32)
        
        self._log = get_logger("security.secrets")
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt a secret value."""
        # Using XOR with key-derived pad (simple but effective for config)
        # For production, use a proper encryption library like cryptography
        data = plaintext.encode()
        key_stream = self._generate_keystream(len(data))
        encrypted = bytes(a ^ b for a, b in zip(data, key_stream))
        return base64.urlsafe_b64encode(encrypted).decode('ascii')
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a secret value."""
        try:
            data = base64.urlsafe_b64decode(ciphertext)
            key_stream = self._generate_keystream(len(data))
            decrypted = bytes(a ^ b for a, b in zip(data, key_stream))
            return decrypted.decode()
        except Exception as e:
            self._log.error(f"Decryption failed: {e}")
            raise ValueError("Invalid ciphertext")
    
    def _generate_keystream(self, length: int) -> bytes:
        """Generate a keystream for encryption."""
        # Expand key to required length
        result = b''
        counter = 0
        while len(result) < length:
            block = hashlib.sha256(self._key + counter.to_bytes(4, 'big')).digest()
            result += block
            counter += 1
        return result[:length]
    
    def mask_secret(self, value: str, visible_chars: int = 4) -> str:
        """Mask a secret for logging (show only first/last chars)."""
        if not value or len(value) <= visible_chars * 2:
            return '*' * len(value) if value else ''
        
        return f"{value[:visible_chars]}...{value[-visible_chars:]}"


# ==============================================================================
# SECURITY MANAGER (MAIN CLASS)
# ==============================================================================

class SecurityManager:
    """
    Central security manager that coordinates all security components.
    Use this as the main interface for security operations.
    """
    
    _instance: Optional["SecurityManager"] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.rate_limiter = SecurityRateLimiter(self.config)
        self.validator = InputValidator(self.config)
        self.nonce_manager = NonceManager(self.config.nonce_expiration)
        self.hmac_verifier = HMACVerifier.from_env()
        self.secret_manager = SecretManager()
        self._events: List[SecurityEvent] = []
        self._events_lock = threading.Lock()
        self._log = get_logger("security")
    
    @classmethod
    def get_instance(cls, config: Optional[SecurityConfig] = None) -> "SecurityManager":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    def check_request(
        self,
        source: str,
        payload: Optional[bytes] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive request validation.
        
        Args:
            source: Request source identifier (IP, user ID, etc.)
            payload: Optional request payload
        
        Returns:
            Tuple of (is_allowed, error_message)
        """
        # Check rate limit
        if self.config.enable_rate_limiting:
            allowed, reason = self.rate_limiter.check(source)
            if not allowed:
                self._record_event(SecurityEvent(
                    event_type='rate_limit',
                    timestamp=datetime.now(timezone.utc),
                    source=source,
                    details={'reason': reason},
                    severity='warning'
                ))
                return False, reason
        
        # Validate payload size
        if payload and self.config.enable_input_validation:
            valid, error = self.validator.validate_json_size(payload)
            if not valid:
                self._record_event(SecurityEvent(
                    event_type='invalid_input',
                    timestamp=datetime.now(timezone.utc),
                    source=source,
                    details={'error': error, 'size': len(payload)},
                    severity='warning'
                ))
                return False, error
        
        return True, None
    
    def validate_input(self, value: str, context: str = "input") -> str:
        """Validate and sanitize input string."""
        valid, error = self.validator.validate_string(value)
        if not valid:
            self._record_event(SecurityEvent(
                event_type='invalid_input',
                timestamp=datetime.now(timezone.utc),
                source=context,
                details={'error': error},
                severity='warning'
            ))
            raise ValueError(error)
        return self.validator.sanitize_string(value)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: str, algorithm: str = 'sha256') -> str:
        """Hash data using specified algorithm."""
        h = hashlib.new(algorithm)
        h.update(data.encode())
        return h.hexdigest()
    
    def _record_event(self, event: SecurityEvent):
        """Record a security event."""
        with self._events_lock:
            self._events.append(event)
            # Keep only last 1000 events
            if len(self._events) > 1000:
                self._events = self._events[-1000:]
        
        if self.config.log_security_events:
            log_method = {
                'info': self._log.info,
                'warning': self._log.warn,
                'critical': self._log.error
            }.get(event.severity, self._log.info)
            log_method(f"Security event: {event.event_type} from {event.source}")
    
    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        with self._events_lock:
            return [e.to_dict() for e in self._events[-limit:]]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        return {
            'rate_limiter': self.rate_limiter.get_stats(),
            'recent_events': len(self._events),
            'config': {
                'rate_limiting_enabled': self.config.enable_rate_limiting,
                'input_validation_enabled': self.config.enable_input_validation,
                'max_request_size': self.config.max_request_size
            }
        }


# ==============================================================================
# DECORATORS
# ==============================================================================

def rate_limited(source_extractor: Optional[Callable] = None):
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        source_extractor: Function to extract source identifier from args
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = SecurityManager.get_instance()
            
            # Extract source
            if source_extractor:
                source = source_extractor(*args, **kwargs)
            else:
                source = func.__name__
            
            # Check rate limit
            allowed, reason = security.rate_limiter.check(source)
            if not allowed:
                raise RuntimeError(f"Rate limited: {reason}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validated_input(*param_names: str):
    """
    Decorator to validate string parameters.
    
    Args:
        param_names: Names of parameters to validate
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = SecurityManager.get_instance()
            
            # Validate named parameters
            for name in param_names:
                if name in kwargs:
                    value = kwargs[name]
                    if isinstance(value, str):
                        kwargs[name] = security.validate_input(value, f"{func.__name__}.{name}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Get the global security manager instance."""
    return SecurityManager.get_instance(config)


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_hex(32)


def generate_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(48)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time (prevents timing attacks)."""
    return hmac.compare_digest(a.encode(), b.encode())


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Hash a password using PBKDF2.
    
    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    
    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt,
        100000
    )
    return hash_bytes, salt


def verify_password(password: str, hash_bytes: bytes, salt: bytes) -> bool:
    """Verify a password against a hash."""
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, hash_bytes)


# ==============================================================================
# INITIALIZATION
# ==============================================================================

def init_security(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Initialize the security system."""
    log = get_logger("security")
    log.info("Initializing security system...")
    
    manager = get_security_manager(config)
    
    log.info("✓ Security system initialized")
    log.info(f"  Rate limiting: {'enabled' if manager.config.enable_rate_limiting else 'disabled'}")
    log.info(f"  Input validation: {'enabled' if manager.config.enable_input_validation else 'disabled'}")
    log.info(f"  Max request size: {manager.config.max_request_size} bytes")
    
    return manager
