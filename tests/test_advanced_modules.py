"""
Test suite for new v2.1 modules: crypto, backup, errors, telemetry.
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================
# Test Crypto Module
# ============================================================

class TestEd25519:
    """Tests for Ed25519 implementation."""
    
    def test_keypair_generation(self):
        """Test keypair generation."""
        from agent.crypto import Ed25519
        
        private_key, public_key = Ed25519.generate_keypair()
        
        assert len(private_key) == 32
        assert len(public_key) == 32
    
    def test_keypair_deterministic(self):
        """Test keypair generation is deterministic with seed."""
        from agent.crypto import Ed25519
        
        seed = b"test_seed_12345678901234567890"
        
        pk1, pub1 = Ed25519.generate_keypair(seed)
        pk2, pub2 = Ed25519.generate_keypair(seed)
        
        assert pk1 == pk2
        assert pub1 == pub2
    
    def test_sign_and_verify(self):
        """Test signing and verification."""
        from agent.crypto import Ed25519
        
        private_key, public_key = Ed25519.generate_keypair()
        message = b"Hello, Solana!"
        
        signature = Ed25519.sign(private_key, message)
        
        assert len(signature) == 64
        assert Ed25519.verify(public_key, message, signature)
    
    def test_verify_fails_with_wrong_message(self):
        """Test verification fails with wrong message."""
        from agent.crypto import Ed25519
        
        private_key, public_key = Ed25519.generate_keypair()
        message = b"Hello, Solana!"
        wrong_message = b"Wrong message"
        
        signature = Ed25519.sign(private_key, message)
        
        assert not Ed25519.verify(public_key, wrong_message, signature)
    
    def test_verify_fails_with_wrong_key(self):
        """Test verification fails with wrong key."""
        from agent.crypto import Ed25519
        
        private_key1, public_key1 = Ed25519.generate_keypair()
        _, public_key2 = Ed25519.generate_keypair()
        message = b"Hello, Solana!"
        
        signature = Ed25519.sign(private_key1, message)
        
        assert Ed25519.verify(public_key1, message, signature)
        assert not Ed25519.verify(public_key2, message, signature)


class TestBase58:
    """Tests for Base58 encoding."""
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding."""
        from agent.crypto import Base58
        
        data = b"\x00\x01\x02\x03\x04\x05"
        encoded = Base58.encode(data)
        decoded = Base58.decode(encoded)
        
        assert decoded == data
    
    def test_encode_solana_address_format(self):
        """Test Solana address encoding."""
        from agent.crypto import Base58
        
        # 32-byte public key
        public_key = bytes(range(32))
        encoded = Base58.encode(public_key)
        
        # Solana addresses are 32-44 base58 characters
        assert 32 <= len(encoded) <= 44


class TestSolanaKeypair:
    """Tests for SolanaKeypair class."""
    
    def test_generate(self):
        """Test keypair generation."""
        from agent.crypto import SolanaKeypair
        
        keypair = SolanaKeypair.generate()
        
        assert len(keypair.private_key) == 32
        assert len(keypair.public_key) == 32
        assert len(keypair.address) >= 32
    
    def test_from_secret_key(self):
        """Test loading from 64-byte secret key."""
        from agent.crypto import SolanaKeypair
        
        keypair1 = SolanaKeypair.generate()
        secret = keypair1.secret_key
        
        keypair2 = SolanaKeypair.from_secret_key(secret)
        
        assert keypair1.address == keypair2.address
    
    def test_sign_and_verify(self):
        """Test signing and verification."""
        from agent.crypto import SolanaKeypair
        
        keypair = SolanaKeypair.generate()
        message = b"Test message"
        
        signature = keypair.sign(message)
        
        assert keypair.verify(message, signature)
    
    def test_save_and_load(self):
        """Test saving and loading keypair."""
        from agent.crypto import SolanaKeypair
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "keypair.json"
            
            keypair1 = SolanaKeypair.generate()
            keypair1.save(path)
            
            keypair2 = SolanaKeypair.load(path)
            
            assert keypair1.address == keypair2.address


class TestProofSigner:
    """Tests for ProofSigner class."""
    
    def test_sign_proof(self):
        """Test proof signing."""
        from agent.crypto import ProofSigner, SolanaKeypair
        
        keypair = SolanaKeypair.generate()
        signer = ProofSigner(keypair=keypair)
        
        proof = signer.sign_proof(task_id=1, result_hash="abc123")
        
        assert proof.task_id == 1
        assert proof.result_hash == "abc123"
        assert proof.public_key == keypair.address
        assert proof.verify()
    
    def test_sign_message(self):
        """Test message signing."""
        from agent.crypto import ProofSigner, SolanaKeypair
        
        keypair = SolanaKeypair.generate()
        signer = ProofSigner(keypair=keypair)
        
        message = "Test message"
        signature = signer.sign_message(message)
        
        assert signer.verify_message(message, signature)


# ============================================================
# Test Errors Module
# ============================================================

class TestErrorCodes:
    """Tests for error code system."""
    
    def test_error_code_format(self):
        """Test error code format."""
        from agent.errors import ErrorCode
        
        code = ErrorCode.AP_RATE_LIMITED
        
        assert "_" in code.value
        assert code.category == "API"
    
    def test_all_codes_have_category(self):
        """Test all error codes have a category."""
        from agent.errors import ErrorCode
        
        for code in ErrorCode:
            assert code.category is not None


class TestAgentError:
    """Tests for AgentError exception."""
    
    def test_create_error(self):
        """Test creating an error."""
        from agent.errors import AgentError, ErrorCode
        
        error = AgentError(
            message="Test error",
            code=ErrorCode.AP_REQUEST_FAILED,
        )
        
        assert error.message == "Test error"
        assert error.code == ErrorCode.AP_REQUEST_FAILED
        assert "AP_001_001" in str(error)
    
    def test_error_with_cause(self):
        """Test error with cause."""
        from agent.errors import AgentError, ErrorCode
        
        cause = ValueError("Original error")
        error = AgentError(
            message="Wrapped error",
            code=ErrorCode.SY_UNEXPECTED_ERROR,
            cause=cause,
        )
        
        assert error.cause == cause
        assert error.stack_trace is not None
    
    def test_error_context(self):
        """Test error context."""
        from agent.errors import AgentError, ErrorCode, ErrorSeverity
        
        error = AgentError(
            message="Test error",
            code=ErrorCode.SO_RPC_ERROR,
            severity=ErrorSeverity.WARNING,
            details={"rpc_url": "https://example.com"},
        )
        
        context = error.context
        
        assert context.code == ErrorCode.SO_RPC_ERROR
        assert context.severity == ErrorSeverity.WARNING
        assert "rpc_url" in context.details


class TestErrorRegistry:
    """Tests for ErrorRegistry."""
    
    def test_record_error(self):
        """Test recording an error."""
        from agent.errors import ErrorRegistry, AgentError, ErrorCode
        
        registry = ErrorRegistry()
        error = AgentError("Test", code=ErrorCode.AP_SERVER_ERROR)
        
        registry.record(error)
        
        assert len(registry.errors) == 1
        assert registry.counts.get(ErrorCode.AP_SERVER_ERROR.value) == 1
    
    def test_get_stats(self):
        """Test getting error stats."""
        from agent.errors import ErrorRegistry, AgentError, ErrorCode
        
        registry = ErrorRegistry()
        registry.record(AgentError("Error 1", code=ErrorCode.AP_SERVER_ERROR))
        registry.record(AgentError("Error 2", code=ErrorCode.AP_SERVER_ERROR))
        registry.record(AgentError("Error 3", code=ErrorCode.NE_CONNECTION_TIMEOUT))
        
        stats = registry.get_stats()
        
        assert stats["total"] == 3
        assert stats["by_code"][ErrorCode.AP_SERVER_ERROR.value] == 2


# ============================================================
# Test Backup Module
# ============================================================

class TestStateSnapshot:
    """Tests for StateSnapshot class."""
    
    def test_capture_and_restore(self):
        """Test capturing and restoring snapshots."""
        from agent.backup import StateSnapshot
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create test data
            (data_dir / "test.json").write_text('{"key": "value"}')
            
            snapshot = StateSnapshot(data_dir=data_dir)
            name = snapshot.capture("test_snapshot")
            
            # Modify data
            (data_dir / "test.json").write_text('{"key": "modified"}')
            
            # Restore
            assert snapshot.restore(name)
            
            # Verify restored
            data = json.loads((data_dir / "test.json").read_text())
            assert data["key"] == "value"
    
    def test_list_snapshots(self):
        """Test listing snapshots."""
        from agent.backup import StateSnapshot
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "data.json").write_text('{}')
            
            snapshot = StateSnapshot(data_dir=data_dir)
            snapshot.capture("snap1")
            snapshot.capture("snap2")
            
            snapshots = snapshot.list_snapshots()
            
            assert len(snapshots) >= 2


# ============================================================
# Test Telemetry Module
# ============================================================

class TestTraceContext:
    """Tests for TraceContext."""
    
    def test_generate_ids(self):
        """Test ID generation."""
        from agent.telemetry import TraceContext
        
        ctx = TraceContext()
        
        assert len(ctx.trace_id) == 48
        assert len(ctx.span_id) == 16
    
    def test_create_child(self):
        """Test creating child context."""
        from agent.telemetry import TraceContext
        
        parent = TraceContext()
        child = parent.create_child()
        
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        assert child.span_id != parent.span_id
    
    def test_to_header(self):
        """Test W3C header format."""
        from agent.telemetry import TraceContext
        
        ctx = TraceContext(sampled=True)
        header = ctx.to_header()
        
        assert header.startswith("00-")
        assert header.endswith("-01")
    
    def test_from_header(self):
        """Test parsing W3C header."""
        from agent.telemetry import TraceContext
        
        ctx1 = TraceContext()
        header = ctx1.to_header()
        
        ctx2 = TraceContext.from_header(header)
        
        assert ctx2.trace_id == ctx1.trace_id
        assert ctx2.span_id == ctx1.span_id


class TestSpan:
    """Tests for Span class."""
    
    def test_span_creation(self):
        """Test span creation."""
        from agent.telemetry import Span, SpanKind
        
        span = Span(
            trace_id="abc123",
            span_id="def456",
            name="test_operation",
            kind=SpanKind.INTERNAL,
        )
        
        assert span.name == "test_operation"
        assert not span.is_ended
    
    def test_span_duration(self):
        """Test span duration calculation."""
        from agent.telemetry import Span
        
        span = Span(
            trace_id="abc",
            span_id="def",
            name="test",
        )
        
        time.sleep(0.1)
        span.end()
        
        assert span.duration_ms >= 100
    
    def test_span_events(self):
        """Test adding events to span."""
        from agent.telemetry import Span
        
        span = Span(trace_id="abc", span_id="def", name="test")
        span.add_event("checkpoint", {"step": 1})
        span.add_event("error", {"message": "test error"})
        
        assert len(span.events) == 2


class TestTracer:
    """Tests for Tracer class."""
    
    def test_start_span_context_manager(self):
        """Test span as context manager."""
        from agent.telemetry import Tracer, SpanStatus
        
        tracer = Tracer("test-service")
        
        with tracer.start_span("test_operation") as span:
            span.set_attribute("key", "value")
        
        assert span.is_ended
        assert span.status == SpanStatus.OK
    
    def test_span_error_handling(self):
        """Test span captures errors."""
        from agent.telemetry import Tracer, SpanStatus
        
        tracer = Tracer("test-service")
        
        with pytest.raises(ValueError):
            with tracer.start_span("failing_operation") as span:
                raise ValueError("Test error")
        
        assert span.status == SpanStatus.ERROR
        assert len(span.events) > 0


class TestProfiler:
    """Tests for Profiler class."""
    
    def test_profile_code_block(self):
        """Test profiling a code block."""
        from agent.telemetry import Profiler
        
        profiler = Profiler()
        
        with profiler.profile("test_function"):
            time.sleep(0.1)
        
        assert len(profiler.samples) == 1
        assert profiler.samples[0].duration_ms >= 100
    
    def test_get_stats(self):
        """Test getting profiler stats."""
        from agent.telemetry import Profiler
        
        profiler = Profiler()
        
        for _ in range(10):
            with profiler.profile("test_func"):
                time.sleep(0.01)
        
        stats = profiler.get_stats("test_func")
        
        assert stats["count"] == 10
        assert stats["avg_ms"] >= 10


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for new modules."""
    
    def test_signed_proof_workflow(self):
        """Test complete proof signing workflow."""
        from agent.crypto import ProofSigner, ProofHashchain, SolanaKeypair
        
        # Generate keypair
        keypair = SolanaKeypair.generate()
        signer = ProofSigner(keypair=keypair)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_file = Path(tmpdir) / "hashchain.json"
            hashchain = ProofHashchain(chain_file=chain_file)
            
            # Sign a proof
            result = "This is the solution"
            result_hash = hashlib.sha256(result.encode()).hexdigest()
            proof = signer.sign_proof(task_id=1, result_hash=result_hash)
            
            # Add to hashchain
            block = hashchain.add_proof(proof)
            
            # Verify
            assert proof.verify()
            assert hashchain.verify_chain()
            assert block.index == 0
    
    def test_error_to_telemetry_flow(self):
        """Test error recording with telemetry."""
        from agent.errors import AgentError, ErrorCode, get_error_registry
        from agent.telemetry import get_telemetry, SpanStatus
        
        telemetry = get_telemetry()
        registry = get_error_registry()
        
        initial_count = len(registry.errors)
        
        with telemetry.tracer.start_span("test_with_error") as span:
            try:
                raise AgentError("Test error", code=ErrorCode.TA_PROCESSING_FAILED)
            except AgentError as e:
                registry.record(e)
                span.set_status(SpanStatus.ERROR, str(e))
        
        assert len(registry.errors) > initial_count
        assert span.status == SpanStatus.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
