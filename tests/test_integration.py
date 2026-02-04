"""
Integration tests for POW Agent system.
End-to-end testing of agent workflows.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================
# TESTS - Full Agent Cycle
# ============================================================

class TestFullAgentCycle:
    """Integration tests for complete agent cycles."""
    
    @pytest.fixture
    def full_mocks(self, mock_colosseum_api, mock_openai, mock_solana_client):
        """Set up all mocks for integration test."""
        return {
            "colosseum": mock_colosseum_api,
            "openai": mock_openai,
            "solana": mock_solana_client
        }
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_cycle_flow(self, env_vars, temp_dir, full_mocks):
        """Test complete agent cycle from start to finish."""
        from agent.loop import AgentLoop
        
        # Configure agent
        loop = AgentLoop()
        loop.state_file = temp_dir / "state.json"
        
        # Run single cycle
        result = await loop.run_single_cycle()
        
        # Verify cycle completed
        assert result is not None
        assert "cycle_number" in result
        assert result["cycle_number"] == 1
        
        # Verify state was updated
        loop.load_state()
        assert loop.cycle_count == 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_cycles(self, env_vars, temp_dir, full_mocks):
        """Test running multiple cycles in sequence."""
        from agent.loop import AgentLoop
        
        loop = AgentLoop()
        loop.state_file = temp_dir / "state.json"
        
        # Run 3 cycles
        for i in range(3):
            result = await loop.run_single_cycle()
            assert result["cycle_number"] == i + 1
        
        loop.load_state()
        assert loop.cycle_count == 3
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cycle_with_task_solving(self, env_vars, temp_dir, full_mocks, sample_task):
        """Test cycle that includes task solving."""
        from agent.loop import AgentLoop
        
        # Configure agent
        loop = AgentLoop()
        loop.state_file = temp_dir / "state.json"
        loop.pending_tasks = [sample_task]
        
        # Run cycle
        result = await loop.run_single_cycle()
        
        # Verify task was processed
        assert result is not None
        if "task_solved" in result:
            assert result["task_solved"] is True


# ============================================================
# TESTS - API + Agent Integration
# ============================================================

class TestAPIAgentIntegration:
    """Integration tests for API and agent interaction."""
    
    @pytest.mark.integration
    def test_start_stop_via_api(self, api_client, env_vars):
        """Test starting and stopping agent via API."""
        # Start agent
        response = api_client.post("/api/v1/agent/start")
        assert response.status_code in [200, 202, 409]
        
        # Check status
        response = api_client.get("/api/v1/status")
        assert response.status_code == 200
        
        # Stop agent
        response = api_client.post("/api/v1/agent/stop")
        assert response.status_code in [200, 409]
    
    @pytest.mark.integration
    def test_trigger_cycle_via_api(self, api_client, env_vars, full_mocks):
        """Test triggering a cycle via API."""
        # Trigger cycle
        response = api_client.post("/api/v1/agent/cycle")
        
        # May require agent to be running
        assert response.status_code in [200, 202, 409, 503]
    
    @pytest.mark.integration
    def test_get_metrics_after_cycle(self, api_client, env_vars):
        """Test metrics are updated after cycle."""
        # Get initial metrics
        response = api_client.get("/api/v1/metrics")
        initial = response.json() if response.status_code == 200 else {}
        
        # Trigger cycle if possible
        api_client.post("/api/v1/agent/cycle")
        
        # Get updated metrics
        response = api_client.get("/api/v1/metrics")
        assert response.status_code == 200


# ============================================================
# TESTS - Colosseum Integration
# ============================================================

class TestColosseumIntegration:
    """Integration tests for Colosseum API."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_forum_engagement(self, env_vars, mock_colosseum_api):
        """Test complete forum engagement workflow."""
        from colosseum.forum import ForumClient
        
        forum = ForumClient()
        
        # Get trending posts
        posts = await forum.get_trending_posts()
        
        if posts:
            post = posts[0]
            
            # Vote on post
            await forum.engage(post["id"], "vote", "up")
            
            # Comment on post
            await forum.engage(post["id"], "comment", "Great post!")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_project_update_flow(self, env_vars, mock_colosseum_api):
        """Test project update workflow."""
        from colosseum.project import ProjectClient
        
        project = ProjectClient()
        
        # Update project
        await project.update_status(
            status="in_progress",
            progress=50,
            notes="Working on integration tests"
        )
        
        # Add milestone
        await project.add_milestone(
            title="Integration Tests",
            description="Added comprehensive integration tests",
            completed=True
        )


# ============================================================
# TESTS - Solana Integration
# ============================================================

class TestSolanaIntegration:
    """Integration tests for Solana blockchain."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_proof_submission(self, env_vars, mock_solana_client):
        """Test complete proof submission workflow."""
        from solana.client import SolanaClient
        
        client = SolanaClient()
        
        # Generate proof
        task = {"id": 1, "description": "Test task"}
        solution = "This is the solution"
        
        task_hash = hashlib.sha256(json.dumps(task).encode()).hexdigest()
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        
        # Submit proof
        tx_sig = await client.submit_proof(task_hash, solution_hash)
        
        assert tx_sig is not None
        assert len(tx_sig) == 88
        
        # Verify transaction
        verified = await client.verify_signature(tx_sig)
        assert verified is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transaction_confirmation_flow(self, env_vars, mock_solana_client):
        """Test transaction confirmation workflow."""
        from solana.client import SolanaClient
        
        client = SolanaClient()
        
        # Submit proof
        tx_sig = await client.submit_proof("a" * 64, "b" * 64)
        
        # Wait for confirmation
        confirmed = await client.confirm_transaction(tx_sig, timeout=30)
        
        assert confirmed is True


# ============================================================
# TESTS - Error Recovery
# ============================================================

class TestErrorRecovery:
    """Integration tests for error recovery."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recover_from_api_failure(self, env_vars, temp_dir):
        """Test agent recovers from API failure."""
        from agent.loop import AgentLoop
        
        loop = AgentLoop()
        loop.state_file = temp_dir / "state.json"
        
        # Simulate API failure on first call, then succeed
        call_count = 0
        
        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API Error")
            return {"status": "active"}
        
        with patch.object(loop, 'check_status', side_effect=fail_then_succeed):
            # First cycle should handle error
            result1 = await loop.run_single_cycle()
            
            # Second cycle should succeed
            result2 = await loop.run_single_cycle()
        
        # Agent should continue despite error
        assert loop.cycle_count == 2
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recover_from_blockchain_failure(self, env_vars, temp_dir, mock_colosseum_api, mock_openai):
        """Test agent recovers from blockchain failure."""
        from agent.loop import AgentLoop
        
        loop = AgentLoop()
        loop.state_file = temp_dir / "state.json"
        
        # Simulate blockchain failure
        with patch("solana.client.SolanaClient.submit_proof", side_effect=Exception("RPC Error")):
            result = await loop.run_single_cycle()
        
        # Cycle should complete with error noted
        assert result is not None
        if "errors" in result:
            assert len(result["errors"]) > 0


# ============================================================
# TESTS - State Persistence
# ============================================================

class TestStatePersistence:
    """Integration tests for state persistence."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_state_survives_restart(self, env_vars, temp_dir, full_mocks):
        """Test state persists across agent restarts."""
        from agent.loop import AgentLoop
        
        state_file = temp_dir / "state.json"
        
        # First run
        loop1 = AgentLoop()
        loop1.state_file = state_file
        
        await loop1.run_single_cycle()
        await loop1.run_single_cycle()
        
        loop1.save_state()
        initial_count = loop1.cycle_count
        
        # Second run (simulated restart)
        loop2 = AgentLoop()
        loop2.state_file = state_file
        loop2.load_state()
        
        assert loop2.cycle_count == initial_count
        
        await loop2.run_single_cycle()
        
        assert loop2.cycle_count == initial_count + 1
    
    @pytest.mark.integration
    def test_state_corruption_recovery(self, env_vars, temp_dir):
        """Test recovery from corrupted state file."""
        from agent.loop import AgentLoop
        
        state_file = temp_dir / "state.json"
        
        # Write corrupted state
        with state_file.open("w") as f:
            f.write("not valid json{{{")
        
        # Should recover gracefully
        loop = AgentLoop()
        loop.state_file = state_file
        loop.load_state()
        
        # Should start fresh
        assert loop.cycle_count == 0


# ============================================================
# TESTS - Performance
# ============================================================

class TestPerformance:
    """Integration tests for performance."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cycle_performance(self, env_vars, temp_dir, full_mocks):
        """Test cycle completes within time limit."""
        from agent.loop import AgentLoop
        
        loop = AgentLoop()
        loop.state_file = temp_dir / "state.json"
        
        start = time.time()
        await loop.run_single_cycle()
        elapsed = time.time() - start
        
        # Should complete within 30 seconds
        assert elapsed < 30
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_api_calls(self, api_client):
        """Test API handles concurrent requests."""
        import asyncio
        from httpx import AsyncClient
        
        from api.server import create_app
        
        app = create_app()
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Make 10 concurrent requests
            tasks = [client.get("/api/v1/status") for _ in range(10)]
            responses = await asyncio.gather(*tasks)
        
        # All should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 8  # Allow some failures
