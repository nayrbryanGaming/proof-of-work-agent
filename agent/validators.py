"""
Validators for agent operations.
Provides input validation, schema validation, and business logic validation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from agent.logger import get_logger


class ValidationError(Exception):
    """Validation error with details."""
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code or "VALIDATION_ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error': self.code,
            'message': self.message,
            'field': self.field,
        }


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def add_error(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        self.errors.append(ValidationError(message, field, code))
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def merge(self, other: "ValidationResult"):
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
    
    def raise_if_invalid(self):
        if not self.is_valid:
            raise ValidationError(
                f"Validation failed: {len(self.errors)} error(s)",
                code="MULTI_VALIDATION_ERROR"
            )


# ============================================================
# String Validators
# ============================================================

def validate_non_empty(value: Any, field_name: str) -> ValidationResult:
    """Validate that a value is not empty."""
    result = ValidationResult()
    
    if value is None:
        result.add_error(f"{field_name} is required", field_name, "REQUIRED")
    elif isinstance(value, str) and not value.strip():
        result.add_error(f"{field_name} cannot be empty", field_name, "EMPTY")
    elif isinstance(value, (list, dict)) and len(value) == 0:
        result.add_error(f"{field_name} cannot be empty", field_name, "EMPTY")
    
    return result


def validate_string_length(
    value: str, 
    field_name: str, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None
) -> ValidationResult:
    """Validate string length."""
    result = ValidationResult()
    
    if not isinstance(value, str):
        result.add_error(f"{field_name} must be a string", field_name, "TYPE_ERROR")
        return result
    
    length = len(value)
    
    if min_length is not None and length < min_length:
        result.add_error(
            f"{field_name} must be at least {min_length} characters",
            field_name,
            "MIN_LENGTH"
        )
    
    if max_length is not None and length > max_length:
        result.add_error(
            f"{field_name} must be at most {max_length} characters",
            field_name,
            "MAX_LENGTH"
        )
    
    return result


def validate_pattern(value: str, field_name: str, pattern: str, description: str) -> ValidationResult:
    """Validate string matches pattern."""
    result = ValidationResult()
    
    if not isinstance(value, str):
        result.add_error(f"{field_name} must be a string", field_name, "TYPE_ERROR")
        return result
    
    if not re.match(pattern, value):
        result.add_error(f"{field_name} {description}", field_name, "PATTERN")
    
    return result


# ============================================================
# URL Validators
# ============================================================

def validate_url(value: str, field_name: str, require_https: bool = False) -> ValidationResult:
    """Validate URL format."""
    result = ValidationResult()
    
    if not isinstance(value, str):
        result.add_error(f"{field_name} must be a string", field_name, "TYPE_ERROR")
        return result
    
    try:
        parsed = urlparse(value)
        
        if not parsed.scheme:
            result.add_error(f"{field_name} must have a scheme (http/https)", field_name, "URL_SCHEME")
        elif require_https and parsed.scheme != "https":
            result.add_error(f"{field_name} must use HTTPS", field_name, "URL_HTTPS")
        elif parsed.scheme not in ("http", "https"):
            result.add_error(f"{field_name} must be HTTP or HTTPS", field_name, "URL_SCHEME")
        
        if not parsed.netloc:
            result.add_error(f"{field_name} must have a host", field_name, "URL_HOST")
            
    except Exception as e:
        result.add_error(f"{field_name} is not a valid URL: {e}", field_name, "URL_INVALID")
    
    return result


# ============================================================
# Solana Validators
# ============================================================

BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def validate_base58(value: str, field_name: str) -> ValidationResult:
    """Validate base58 string."""
    result = ValidationResult()
    
    if not isinstance(value, str):
        result.add_error(f"{field_name} must be a string", field_name, "TYPE_ERROR")
        return result
    
    if not value:
        result.add_error(f"{field_name} cannot be empty", field_name, "EMPTY")
        return result
    
    invalid_chars = set(value) - BASE58_ALPHABET
    if invalid_chars:
        result.add_error(
            f"{field_name} contains invalid base58 characters: {invalid_chars}",
            field_name,
            "BASE58_INVALID"
        )
    
    return result


def validate_solana_address(value: str, field_name: str = "address") -> ValidationResult:
    """Validate Solana address (public key)."""
    result = validate_base58(value, field_name)
    
    if result.is_valid:
        # Solana addresses are 32 bytes, which is 43-44 characters in base58
        if len(value) < 32 or len(value) > 44:
            result.add_error(
                f"{field_name} has invalid length for Solana address",
                field_name,
                "SOLANA_ADDRESS_LENGTH"
            )
    
    return result


def validate_solana_signature(value: str, field_name: str = "signature") -> ValidationResult:
    """Validate Solana transaction signature."""
    result = validate_base58(value, field_name)
    
    if result.is_valid:
        # Solana signatures are 64 bytes, which is 87-88 characters in base58
        if len(value) < 80 or len(value) > 90:
            result.add_error(
                f"{field_name} has invalid length for Solana signature",
                field_name,
                "SOLANA_SIGNATURE_LENGTH"
            )
    
    return result


def validate_program_id(value: str, field_name: str = "program_id") -> ValidationResult:
    """Validate Solana program ID."""
    return validate_solana_address(value, field_name)


# ============================================================
# Hash Validators
# ============================================================

def validate_sha256_hash(value: str, field_name: str = "hash") -> ValidationResult:
    """Validate SHA256 hash."""
    result = ValidationResult()
    
    if not isinstance(value, str):
        result.add_error(f"{field_name} must be a string", field_name, "TYPE_ERROR")
        return result
    
    if not re.match(r'^[a-fA-F0-9]{64}$', value):
        result.add_error(
            f"{field_name} must be a valid SHA256 hash (64 hex characters)",
            field_name,
            "SHA256_INVALID"
        )
    
    return result


def compute_hash(data: Union[str, bytes]) -> str:
    """Compute SHA256 hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def verify_hash(data: Union[str, bytes], expected_hash: str) -> bool:
    """Verify data matches expected hash."""
    return compute_hash(data) == expected_hash.lower()


# ============================================================
# API Response Validators
# ============================================================

def validate_api_response(
    response: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
    field_name: str = "response"
) -> ValidationResult:
    """Validate API response structure."""
    result = ValidationResult()
    
    if not isinstance(response, dict):
        result.add_error(f"{field_name} must be an object", field_name, "TYPE_ERROR")
        return result
    
    if required_fields:
        for field in required_fields:
            if field not in response:
                result.add_error(
                    f"Missing required field: {field}",
                    f"{field_name}.{field}",
                    "MISSING_FIELD"
                )
    
    return result


def validate_status_response(response: Dict[str, Any]) -> ValidationResult:
    """Validate agent status response."""
    result = validate_api_response(response, ["active"], "status")
    
    if result.is_valid:
        if not isinstance(response.get("active"), bool):
            result.add_error(
                "active must be a boolean",
                "status.active",
                "TYPE_ERROR"
            )
    
    return result


def validate_forum_post(post: Dict[str, Any]) -> ValidationResult:
    """Validate forum post structure."""
    result = validate_api_response(
        post,
        ["id", "title"],
        "post"
    )
    
    if result.is_valid:
        if not isinstance(post.get("id"), (int, str)):
            result.add_error("id must be a number or string", "post.id", "TYPE_ERROR")
        if not isinstance(post.get("title"), str):
            result.add_error("title must be a string", "post.title", "TYPE_ERROR")
    
    return result


# ============================================================
# Task Validators
# ============================================================

def validate_task(task: Dict[str, Any]) -> ValidationResult:
    """Validate task structure."""
    result = validate_api_response(task, ["id", "description"], "task")
    
    if result.is_valid:
        if not isinstance(task.get("id"), (int, str)):
            result.add_error("id must be a number or string", "task.id", "TYPE_ERROR")
        
        desc = task.get("description", "")
        if not isinstance(desc, str):
            result.add_error("description must be a string", "task.description", "TYPE_ERROR")
        elif len(desc) < 10:
            result.add_error(
                "description must be at least 10 characters",
                "task.description",
                "MIN_LENGTH"
            )
    
    return result


# ============================================================
# Comment Validators
# ============================================================

def validate_forum_comment(comment: str) -> ValidationResult:
    """Validate forum comment."""
    result = ValidationResult()
    
    if not isinstance(comment, str):
        result.add_error("Comment must be a string", "comment", "TYPE_ERROR")
        return result
    
    comment = comment.strip()
    
    if len(comment) < 10:
        result.add_error(
            "Comment must be at least 10 characters",
            "comment",
            "MIN_LENGTH"
        )
    
    if len(comment) > 400:
        result.add_error(
            "Comment must be at most 400 characters",
            "comment",
            "MAX_LENGTH"
        )
    
    # Check for spam patterns
    spam_patterns = [
        r'(.)\1{5,}',  # Repeated characters
        r'(https?://[^\s]+\s*){3,}',  # Multiple URLs
        r'^[A-Z\s!?]+$',  # All caps
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, comment):
            result.add_warning("Comment may be flagged as spam")
            break
    
    return result


# ============================================================
# Configuration Validators
# ============================================================

def validate_config(config: Any) -> ValidationResult:
    """Validate agent configuration."""
    result = ValidationResult()
    
    # Check API key
    api_key = getattr(config, 'colosseum_api_key', None) or getattr(config, 'colosseum', {}).get('api_key')
    if not api_key:
        result.add_error("COLOSSEUM_API_KEY is required", "config.colosseum_api_key", "REQUIRED")
    
    # Check OpenAI key
    openai_key = getattr(config, 'openai_api_key', None) or getattr(config, 'openai', {}).get('api_key')
    if not openai_key:
        result.add_error("OPENAI_API_KEY is required", "config.openai_api_key", "REQUIRED")
    
    # Check wallet session
    wallet = getattr(config, 'agentwallet_session', None)
    if not wallet:
        result.add_error("AGENTWALLET_SESSION is required", "config.agentwallet_session", "REQUIRED")
    
    # Check program ID
    program_id = getattr(config, 'program_id', None) or getattr(config, 'solana', {}).get('program_id')
    if not program_id:
        result.add_warning("PROGRAM_ID is not set - Solana operations will fail")
    elif program_id:
        pid_result = validate_program_id(program_id)
        result.merge(pid_result)
    
    # Check RPC URL
    rpc_url = getattr(config, 'solana_rpc', None) or getattr(config, 'solana', {}).get('rpc_url')
    if rpc_url:
        url_result = validate_url(rpc_url, "solana_rpc")
        result.merge(url_result)
    
    return result


# ============================================================
# Utility Functions
# ============================================================

def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize a string for safe storage/display."""
    if not isinstance(value, str):
        value = str(value)
    
    # Remove control characters except newlines and tabs
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
    
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length - 3] + "..."
    
    return value


def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize sensitive data for logging."""
    sensitive_keys = {'api_key', 'secret', 'password', 'token', 'session', 'key'}
    
    def mask_value(v: Any) -> Any:
        if isinstance(v, str) and len(v) > 8:
            return v[:4] + "****" + v[-4:]
        return "****"
    
    result = {}
    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys):
            result[k] = mask_value(v)
        elif isinstance(v, dict):
            result[k] = sanitize_log_data(v)
        else:
            result[k] = v
    
    return result
