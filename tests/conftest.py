"""
Test configuration and fixtures.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# FIXTURES - Environment
# ============================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def env_vars():
    """Set up test environment variables."""
    original = dict(os.environ)
    
    os.environ.update({
        "COLOSSEUM_API_KEY": "test_api_key_12345",
        "OPENAI_API_KEY": "test_openai_key_12345",
        "AGENTWALLET_SESSION": "test_wallet_session",
        "SOLANA_RPC": "https://api.devnet.solana.com",
        "PROGRAM_ID": "Test111111111111111111111111111111111111111"
    })
    
    yield os.environ
    
    os.environ.clear()
    os.environ.update(original)


# ============================================================
# FIXTURES - API Client
# ============================================================

@pytest.fixture
def api_client():
    """Create test client for FastAPI."""
    from api.server import create_app
    
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_api_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for FastAPI."""
    from api.server import create_app
    
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ============================================================
# FIXTURES - Mock Services
# ============================================================

@pytest.fixture
def mock_colosseum_api():
    """Mock Colosseum API responses."""
    with patch("colosseum.api.ColosseumAPI") as mock:
        api = mock.return_value
        
        api.get_agent_status.return_value = {
            "status": "active",
            "nextSteps": ["engage_forum", "solve_task"]
        }
        
        api.get_hot_posts.return_value = [
            {"id": 1, "title": "Test Post", "tags": ["ai", "solana"]},
            {"id": 2, "title": "Another Post", "tags": ["defi"]}
        ]
        
        api.vote_post.return_value = {"success": True}
        api.comment_post.return_value = {"success": True, "comment_id": 123}
        api.update_project.return_value = {"success": True}
        
        yield api


@pytest.fixture
def mock_openai():
    """Mock OpenAI API responses."""
    with patch("agent.decision._post_openai") as mock:
        mock.return_value = {
            "choices": [{
                "message": {
                    "content": "This is a test response from the AI model."
                }
            }]
        }
        yield mock


@pytest.fixture
def mock_solana_client():
    """Mock Solana client."""
    with patch("solana.client.SolanaClient") as mock:
        client = mock.return_value
        
        client.submit_proof.return_value = "5" + "x" * 87  # Mock TX signature
        client.verify_signature.return_value = True
        client.get_balance.return_value = 1.5
        
        yield client


# ============================================================
# FIXTURES - Data
# ============================================================

@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return {
        "id": 1,
        "description": "Summarize the key features of Solana blockchain",
        "category": "research",
        "difficulty": "medium"
    }


@pytest.fixture
def sample_tasks():
    """List of sample tasks."""
    return [
        {"id": 1, "description": "Task 1", "category": "general", "difficulty": "easy"},
        {"id": 2, "description": "Task 2", "category": "research", "difficulty": "medium"},
        {"id": 3, "description": "Task 3", "category": "development", "difficulty": "hard"}
    ]


@pytest.fixture
def sample_forum_post():
    """Sample forum post."""
    return {
        "id": 123,
        "title": "Building an AI Agent on Solana",
        "content": "I'm building an autonomous AI agent for the hackathon...",
        "tags": ["ai", "solana", "agent"],
        "author": "developer123",
        "votes": 10,
        "comments": []
    }


@pytest.fixture
def sample_cycle_result():
    """Sample cycle result."""
    return {
        "cycle_number": 1,
        "started_at": "2025-02-05T10:00:00Z",
        "completed_at": "2025-02-05T10:01:30Z",
        "duration": 90.5,
        "heartbeat_synced": True,
        "status_checked": True,
        "forum_engaged": True,
        "task_solved": True,
        "task_hash": "abc123" * 10,
        "solana_tx": "5" + "x" * 87,
        "project_updated": True,
        "errors": []
    }


# ============================================================
# FIXTURES - State
# ============================================================

@pytest.fixture
def state_file(temp_dir):
    """Create temporary state file."""
    state_path = temp_dir / "state.json"
    initial_state = {
        "version": "1.0.0",
        "agent_id": "test_agent_123",
        "current_cycle": 0,
        "running": False,
        "metrics": {
            "total_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "tasks_solved": 0
        },
        "recent_cycles": []
    }
    
    with state_path.open("w") as f:
        json.dump(initial_state, f)
    
    yield state_path


@pytest.fixture
def tasks_file(temp_dir, sample_tasks):
    """Create temporary tasks file."""
    tasks_path = temp_dir / "sample_tasks.json"
    
    with tasks_path.open("w") as f:
        json.dump(sample_tasks, f)
    
    yield tasks_path


# ============================================================
# FIXTURES - Logging
# ============================================================

@pytest.fixture
def capture_logs(temp_dir):
    """Capture logs to file."""
    log_path = temp_dir / "test.log"
    
    from agent.logger import setup_logger
    # setup_logger(log_path)  # Would need implementation
    
    yield log_path


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def assert_valid_hash(hash_str: str, length: int = 64):
    """Assert string is valid hex hash."""
    assert len(hash_str) == length
    assert all(c in "0123456789abcdef" for c in hash_str.lower())


def assert_valid_solana_tx(tx_sig: str):
    """Assert string is valid Solana transaction signature."""
    assert len(tx_sig) == 88
    # Base58 characters
    assert all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in tx_sig)


# ============================================================
# PYTEST CONFIGURATION
# ============================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
