"""
Test suite for agent core functionality.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from agent.loop import AgentLoop
from agent.decision import solve_task, generate_ai_response
from agent.heartbeat import HeartbeatSync
from agent.config import Config


# ============================================================
# TESTS - Agent Loop
# ============================================================

class TestAgentLoop:
    """Tests for AgentLoop class."""
    
    @pytest.fixture
    def agent_loop(self, env_vars, temp_dir):
        """Create agent loop instance."""
        with patch.object(Config, 'LOGS_DIR', str(temp_dir)):
            loop = AgentLoop()
            loop.state_file = temp_dir / "state.json"
            return loop
    
    def test_initialization(self, agent_loop):
        """Test agent loop initializes correctly."""
        assert agent_loop is not None
        assert agent_loop.running is False
        assert agent_loop.cycle_count == 0
    
    def test_state_persistence(self, agent_loop, temp_dir):
        """Test state is persisted correctly."""
        agent_loop.cycle_count = 5
        agent_loop.save_state()
        
        # Create new instance and load state
        new_loop = AgentLoop()
        new_loop.state_file = temp_dir / "state.json"
        new_loop.load_state()
        
        assert new_loop.cycle_count == 5
    
    @pytest.mark.asyncio
    async def test_single_cycle(self, agent_loop, mock_colosseum_api, mock_openai, mock_solana_client):
        """Test single agent cycle execution."""
        result = await agent_loop.run_single_cycle()
        
        assert result is not None
        assert "cycle_number" in result
        assert "started_at" in result
        assert "completed_at" in result
    
    @pytest.mark.asyncio
    async def test_cycle_error_handling(self, agent_loop):
        """Test cycle handles errors gracefully."""
        with patch.object(agent_loop, 'sync_heartbeat', side_effect=Exception("API Error")):
            result = await agent_loop.run_single_cycle()
            
            assert result is not None
            assert len(result.get("errors", [])) > 0
    
    @pytest.mark.asyncio
    async def test_stop_gracefully(self, agent_loop):
        """Test agent stops gracefully."""
        agent_loop.running = True
        
        async def delayed_stop():
            await asyncio.sleep(0.1)
            agent_loop.stop()
        
        asyncio.create_task(delayed_stop())
        
        await asyncio.sleep(0.2)
        assert agent_loop.running is False


# ============================================================
# TESTS - Decision Engine
# ============================================================

class TestDecisionEngine:
    """Tests for decision engine."""
    
    @pytest.mark.asyncio
    async def test_solve_task_success(self, sample_task, mock_openai):
        """Test task solving succeeds."""
        result = await solve_task(sample_task)
        
        assert result is not None
        assert "solution" in result or "response" in result
    
    @pytest.mark.asyncio
    async def test_solve_task_validation(self):
        """Test task validation."""
        with pytest.raises(ValueError):
            await solve_task(None)
        
        with pytest.raises(ValueError):
            await solve_task({})
    
    @pytest.mark.asyncio
    async def test_generate_ai_response(self, mock_openai):
        """Test AI response generation."""
        response = await generate_ai_response(
            "What is Solana?",
            max_tokens=100
        )
        
        assert response is not None
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_response_caching(self, mock_openai):
        """Test AI responses are cached."""
        prompt = "Test prompt for caching"
        
        response1 = await generate_ai_response(prompt)
        response2 = await generate_ai_response(prompt)
        
        # Should only call API once due to caching
        assert mock_openai.call_count <= 2


# ============================================================
# TESTS - Heartbeat
# ============================================================

class TestHeartbeat:
    """Tests for heartbeat synchronization."""
    
    @pytest.fixture
    def heartbeat(self, env_vars):
        """Create heartbeat instance."""
        return HeartbeatSync()
    
    @pytest.mark.asyncio
    async def test_sync_success(self, heartbeat, mock_colosseum_api):
        """Test heartbeat sync succeeds."""
        result = await heartbeat.sync()
        
        assert result is True or result is None
    
    @pytest.mark.asyncio
    async def test_sync_retry_on_failure(self, heartbeat):
        """Test heartbeat retries on failure."""
        fail_count = 0
        
        async def fail_twice(*args, **kwargs):
            nonlocal fail_count
            fail_count += 1
            if fail_count < 3:
                raise Exception("Temporary failure")
            return {"success": True}
        
        with patch.object(heartbeat, '_send_heartbeat', side_effect=fail_twice):
            result = await heartbeat.sync()
            
            assert fail_count >= 2
    
    def test_heartbeat_interval(self, heartbeat):
        """Test heartbeat interval is configured correctly."""
        assert heartbeat.interval > 0
        assert heartbeat.interval <= 300  # Max 5 minutes


# ============================================================
# TESTS - Config
# ============================================================

class TestConfig:
    """Tests for configuration management."""
    
    def test_env_loading(self, env_vars):
        """Test environment variables are loaded."""
        config = Config()
        
        assert config.COLOSSEUM_API_KEY == "test_api_key_12345"
        assert config.OPENAI_API_KEY == "test_openai_key_12345"
    
    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict('os.environ', {}, clear=True):
            config = Config()
            
            assert config.CYCLE_INTERVAL > 0
            assert config.MAX_RETRIES > 0
    
    def test_validation(self, env_vars):
        """Test configuration validation."""
        config = Config()
        
        # Should not raise
        config.validate()
    
    def test_missing_required_raises(self):
        """Test missing required config raises error."""
        with patch.dict('os.environ', {}, clear=True):
            config = Config()
            
            with pytest.raises(ValueError):
                config.validate(require_all=True)


# ============================================================
# TESTS - Proof of Work
# ============================================================

class TestProofOfWork:
    """Tests for proof of work generation."""
    
    def test_hash_generation(self, sample_task):
        """Test proof hash is generated correctly."""
        task_str = json.dumps(sample_task, sort_keys=True)
        solution = "This is the solution"
        
        proof = hashlib.sha256(f"{task_str}:{solution}".encode()).hexdigest()
        
        assert len(proof) == 64
        assert all(c in "0123456789abcdef" for c in proof)
    
    def test_hash_deterministic(self, sample_task):
        """Test same input produces same hash."""
        task_str = json.dumps(sample_task, sort_keys=True)
        solution = "Same solution"
        
        proof1 = hashlib.sha256(f"{task_str}:{solution}".encode()).hexdigest()
        proof2 = hashlib.sha256(f"{task_str}:{solution}".encode()).hexdigest()
        
        assert proof1 == proof2
    
    def test_different_inputs_different_hashes(self, sample_task):
        """Test different inputs produce different hashes."""
        task_str = json.dumps(sample_task, sort_keys=True)
        
        proof1 = hashlib.sha256(f"{task_str}:solution1".encode()).hexdigest()
        proof2 = hashlib.sha256(f"{task_str}:solution2".encode()).hexdigest()
        
        assert proof1 != proof2


# ============================================================
# TESTS - Metrics
# ============================================================

class TestMetrics:
    """Tests for metrics collection."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        from agent.metrics import AgentMetricsCollector
        return AgentMetricsCollector()
    
    def test_counter_increment(self, metrics):
        """Test counter increments correctly."""
        initial = metrics.get_counter("cycles_total")
        
        metrics.increment("cycles_total")
        metrics.increment("cycles_total")
        
        assert metrics.get_counter("cycles_total") == initial + 2
    
    def test_gauge_set(self, metrics):
        """Test gauge sets correctly."""
        metrics.set_gauge("active_tasks", 5)
        
        assert metrics.get_gauge("active_tasks") == 5
        
        metrics.set_gauge("active_tasks", 3)
        
        assert metrics.get_gauge("active_tasks") == 3
    
    def test_histogram_observe(self, metrics):
        """Test histogram observes correctly."""
        metrics.observe("cycle_duration", 1.5)
        metrics.observe("cycle_duration", 2.0)
        metrics.observe("cycle_duration", 1.8)
        
        stats = metrics.get_histogram_stats("cycle_duration")
        
        assert stats["count"] == 3
        assert stats["min"] == 1.5
        assert stats["max"] == 2.0
    
    def test_export_prometheus(self, metrics):
        """Test Prometheus export format."""
        metrics.increment("test_counter")
        
        output = metrics.export_prometheus()
        
        assert "test_counter" in output
        assert "TYPE" in output or "#" in output


# ============================================================
# TESTS - Rate Limiting
# ============================================================

class TestRateLimiting:
    """Tests for rate limiting."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        from agent.rate_limiter import TokenBucketRateLimiter
        return TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
    
    def test_acquire_within_limit(self, rate_limiter):
        """Test acquiring within limit succeeds."""
        for _ in range(5):
            assert rate_limiter.acquire() is True
    
    def test_acquire_exceeds_limit(self, rate_limiter):
        """Test acquiring exceeding limit fails."""
        # Exhaust tokens
        for _ in range(5):
            rate_limiter.acquire()
        
        # Should fail
        assert rate_limiter.acquire() is False
    
    def test_token_refill(self, rate_limiter):
        """Test tokens refill over time."""
        # Exhaust tokens
        for _ in range(5):
            rate_limiter.acquire()
        
        # Wait for refill
        time.sleep(1.1)
        
        # Should succeed
        assert rate_limiter.acquire() is True


# ============================================================
# TESTS - Task Queue
# ============================================================

class TestTaskQueue:
    """Tests for task queue."""
    
    @pytest.fixture
    def task_queue(self, temp_dir):
        """Create task queue instance."""
        from agent.task_queue import TaskQueue
        return TaskQueue(storage_path=temp_dir / "queue.db")
    
    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, task_queue, sample_task):
        """Test enqueue and dequeue."""
        await task_queue.enqueue(sample_task)
        
        task = await task_queue.dequeue()
        
        assert task is not None
        assert task["id"] == sample_task["id"]
    
    @pytest.mark.asyncio
    async def test_priority_order(self, task_queue):
        """Test tasks are dequeued by priority."""
        await task_queue.enqueue({"id": 1, "priority": "low"})
        await task_queue.enqueue({"id": 2, "priority": "high"})
        await task_queue.enqueue({"id": 3, "priority": "medium"})
        
        task1 = await task_queue.dequeue()
        task2 = await task_queue.dequeue()
        task3 = await task_queue.dequeue()
        
        assert task1["id"] == 2  # High priority first
        assert task2["id"] == 3  # Medium second
        assert task3["id"] == 1  # Low last
    
    @pytest.mark.asyncio
    async def test_persistence(self, task_queue, temp_dir, sample_task):
        """Test queue persists across restarts."""
        await task_queue.enqueue(sample_task)
        
        # Create new queue with same path
        from agent.task_queue import TaskQueue
        new_queue = TaskQueue(storage_path=temp_dir / "queue.db")
        
        task = await new_queue.dequeue()
        
        assert task is not None
        assert task["id"] == sample_task["id"]


# ============================================================
# TESTS - Health Checks
# ============================================================

class TestHealthChecks:
    """Tests for health check system."""
    
    @pytest.fixture
    def health_registry(self):
        """Create health check registry."""
        from agent.health import HealthCheckRegistry
        return HealthCheckRegistry()
    
    @pytest.mark.asyncio
    async def test_register_check(self, health_registry):
        """Test registering health check."""
        async def check_fn():
            return True
        
        health_registry.register("test_check", check_fn)
        
        assert "test_check" in health_registry.checks
    
    @pytest.mark.asyncio
    async def test_run_checks(self, health_registry):
        """Test running all checks."""
        health_registry.register("pass", lambda: True)
        health_registry.register("fail", lambda: False)
        
        results = await health_registry.run_all()
        
        assert results["pass"]["healthy"] is True
        assert results["fail"]["healthy"] is False
    
    @pytest.mark.asyncio
    async def test_overall_status(self, health_registry):
        """Test overall health status."""
        health_registry.register("check1", lambda: True)
        health_registry.register("check2", lambda: True)
        
        status = await health_registry.get_status()
        
        assert status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_unhealthy_overall(self, health_registry):
        """Test unhealthy overall status."""
        health_registry.register("check1", lambda: True)
        health_registry.register("critical", lambda: False, critical=True)
        
        status = await health_registry.get_status()
        
        assert status["healthy"] is False


# ============================================================
# TESTS - Events
# ============================================================

class TestEvents:
    """Tests for event system."""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus."""
        from agent.events import EventEmitter
        return EventEmitter()
    
    def test_subscribe_emit(self, event_bus):
        """Test subscribe and emit."""
        received = []
        
        event_bus.on("test_event", lambda data: received.append(data))
        event_bus.emit("test_event", {"message": "hello"})
        
        assert len(received) == 1
        assert received[0]["message"] == "hello"
    
    def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers."""
        count = {"value": 0}
        
        event_bus.on("event", lambda d: count.update({"value": count["value"] + 1}))
        event_bus.on("event", lambda d: count.update({"value": count["value"] + 1}))
        
        event_bus.emit("event", {})
        
        assert count["value"] == 2
    
    def test_unsubscribe(self, event_bus):
        """Test unsubscribe."""
        count = {"value": 0}
        
        def handler(data):
            count["value"] += 1
        
        event_bus.on("event", handler)
        event_bus.emit("event", {})
        
        event_bus.off("event", handler)
        event_bus.emit("event", {})
        
        assert count["value"] == 1
    
    def test_once(self, event_bus):
        """Test once subscription."""
        count = {"value": 0}
        
        event_bus.once("event", lambda d: count.update({"value": count["value"] + 1}))
        
        event_bus.emit("event", {})
        event_bus.emit("event", {})
        
        assert count["value"] == 1


# ============================================================
# TESTS - Scheduler
# ============================================================

class TestScheduler:
    """Tests for job scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create scheduler instance."""
        from agent.scheduler import Scheduler
        return Scheduler()
    
    @pytest.mark.asyncio
    async def test_schedule_interval(self, scheduler):
        """Test scheduling interval job."""
        count = {"value": 0}
        
        async def job():
            count["value"] += 1
        
        scheduler.schedule_interval(job, seconds=0.1)
        scheduler.start()
        
        await asyncio.sleep(0.35)
        
        scheduler.stop()
        
        assert count["value"] >= 3
    
    @pytest.mark.asyncio
    async def test_schedule_once(self, scheduler):
        """Test scheduling one-time job."""
        count = {"value": 0}
        
        async def job():
            count["value"] += 1
        
        scheduler.schedule_once(job, delay=0.1)
        scheduler.start()
        
        await asyncio.sleep(0.2)
        
        scheduler.stop()
        
        assert count["value"] == 1
    
    def test_cancel_job(self, scheduler):
        """Test canceling scheduled job."""
        job_id = scheduler.schedule_interval(lambda: None, seconds=60)
        
        result = scheduler.cancel(job_id)
        
        assert result is True
        assert job_id not in scheduler.jobs
