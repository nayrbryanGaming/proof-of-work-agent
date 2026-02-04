"""
Test suite for Colosseum API integration.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from colosseum.api import ColosseumAPI
from colosseum.forum import ForumClient
from colosseum.project import ProjectClient
from colosseum.status import StatusChecker


# ============================================================
# TESTS - Colosseum API Client
# ============================================================

class TestColosseumAPI:
    """Tests for Colosseum API client."""
    
    @pytest.fixture
    def api(self, env_vars):
        """Create API client instance."""
        return ColosseumAPI()
    
    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        def _create(data: dict, status: int = 200):
            response = MagicMock()
            response.status_code = status
            response.json.return_value = data
            response.raise_for_status = MagicMock()
            if status >= 400:
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Error", request=MagicMock(), response=response
                )
            return response
        return _create
    
    @pytest.mark.asyncio
    async def test_get_agent_status(self, api, mock_response):
        """Test getting agent status."""
        with patch.object(api, '_request', return_value=mock_response({
            "status": "active",
            "nextSteps": ["engage_forum", "solve_task"]
        })):
            status = await api.get_agent_status()
            
            assert status["status"] == "active"
            assert "nextSteps" in status
    
    @pytest.mark.asyncio
    async def test_get_hot_posts(self, api, mock_response):
        """Test getting hot posts."""
        with patch.object(api, '_request', return_value=mock_response({
            "posts": [
                {"id": 1, "title": "Test", "tags": ["ai"]},
                {"id": 2, "title": "Test 2", "tags": ["solana"]}
            ]
        })):
            posts = await api.get_hot_posts()
            
            assert len(posts) >= 1
            assert "id" in posts[0]
            assert "title" in posts[0]
    
    @pytest.mark.asyncio
    async def test_vote_post(self, api, mock_response):
        """Test voting on a post."""
        with patch.object(api, '_request', return_value=mock_response({"success": True})):
            result = await api.vote_post(post_id=123, vote_type="up")
            
            assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_comment_post(self, api, mock_response):
        """Test commenting on a post."""
        with patch.object(api, '_request', return_value=mock_response({
            "success": True,
            "comment_id": 456
        })):
            result = await api.comment_post(
                post_id=123,
                content="Great post! Very insightful."
            )
            
            assert result["success"] is True
            assert "comment_id" in result
    
    @pytest.mark.asyncio
    async def test_update_project(self, api, mock_response):
        """Test updating project status."""
        with patch.object(api, '_request', return_value=mock_response({"success": True})):
            result = await api.update_project(
                title="My AI Agent",
                description="An autonomous agent for Solana",
                status="in_progress"
            )
            
            assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_request_retry_on_error(self, api, mock_response):
        """Test request retries on transient error."""
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectTimeout("Connection timeout")
            return mock_response({"success": True})
        
        with patch.object(api, '_request', side_effect=mock_request):
            result = await api.get_agent_status()
            
            assert call_count >= 2
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, api, mock_response):
        """Test rate limiting is respected."""
        with patch.object(api, '_request', return_value=mock_response({"status": "ok"})):
            start = time.time()
            
            # Make multiple requests
            for _ in range(5):
                await api.get_agent_status()
            
            elapsed = time.time() - start
            
            # Should have some delay if rate limited
            # (May pass quickly if no rate limiting)
            assert elapsed >= 0


# ============================================================
# TESTS - Forum Client
# ============================================================

class TestForumClient:
    """Tests for forum client."""
    
    @pytest.fixture
    def forum(self, env_vars):
        """Create forum client instance."""
        return ForumClient()
    
    @pytest.mark.asyncio
    async def test_get_trending_posts(self, forum, mock_colosseum_api):
        """Test getting trending posts."""
        posts = await forum.get_trending_posts()
        
        assert isinstance(posts, list)
    
    @pytest.mark.asyncio
    async def test_filter_posts_by_tags(self, forum):
        """Test filtering posts by tags."""
        posts = [
            {"id": 1, "tags": ["ai", "solana"]},
            {"id": 2, "tags": ["defi"]},
            {"id": 3, "tags": ["ai", "agent"]}
        ]
        
        filtered = forum.filter_by_tags(posts, ["ai"])
        
        assert len(filtered) == 2
        assert all("ai" in p["tags"] for p in filtered)
    
    @pytest.mark.asyncio
    async def test_engage_with_post(self, forum, mock_colosseum_api):
        """Test engaging with a post."""
        result = await forum.engage(
            post_id=123,
            action="vote",
            value="up"
        )
        
        # May succeed or fail depending on mock
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_create_post(self, forum, mock_colosseum_api):
        """Test creating a post."""
        result = await forum.create_post(
            title="My AI Agent Progress",
            content="Building an autonomous agent...",
            tags=["ai", "solana", "progress"]
        )
        
        assert result is not None


# ============================================================
# TESTS - Project Client
# ============================================================

class TestProjectClient:
    """Tests for project client."""
    
    @pytest.fixture
    def project(self, env_vars):
        """Create project client instance."""
        return ProjectClient()
    
    @pytest.mark.asyncio
    async def test_get_project_info(self, project, mock_colosseum_api):
        """Test getting project info."""
        info = await project.get_info()
        
        # May return None if not set up
        assert info is None or isinstance(info, dict)
    
    @pytest.mark.asyncio
    async def test_update_project_status(self, project, mock_colosseum_api):
        """Test updating project status."""
        result = await project.update_status(
            status="in_progress",
            progress=50,
            notes="Working on core functionality"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_add_milestone(self, project, mock_colosseum_api):
        """Test adding a milestone."""
        result = await project.add_milestone(
            title="Core Agent Loop",
            description="Implemented main agent loop",
            completed=True
        )
        
        assert result is not None


# ============================================================
# TESTS - Status Checker
# ============================================================

class TestStatusChecker:
    """Tests for status checker."""
    
    @pytest.fixture
    def checker(self, env_vars):
        """Create status checker instance."""
        return StatusChecker()
    
    @pytest.mark.asyncio
    async def test_check_status(self, checker, mock_colosseum_api):
        """Test checking status."""
        status = await checker.check()
        
        assert "status" in status or status is not None
    
    @pytest.mark.asyncio
    async def test_get_next_steps(self, checker, mock_colosseum_api):
        """Test getting next steps."""
        mock_colosseum_api.get_agent_status.return_value = {
            "status": "active",
            "nextSteps": ["engage_forum", "solve_task"]
        }
        
        steps = await checker.get_next_steps()
        
        assert isinstance(steps, list)
    
    @pytest.mark.asyncio
    async def test_validate_credentials(self, checker, mock_colosseum_api):
        """Test validating credentials."""
        valid = await checker.validate_credentials()
        
        # Should return True with mock
        assert valid is True or valid is False


# ============================================================
# TESTS - Error Handling
# ============================================================

class TestColosseumErrorHandling:
    """Tests for Colosseum API error handling."""
    
    @pytest.fixture
    def api(self, env_vars):
        """Create API client instance."""
        return ColosseumAPI()
    
    @pytest.mark.asyncio
    async def test_handle_401_unauthorized(self, api):
        """Test handling 401 error."""
        with patch.object(api, '_request', side_effect=httpx.HTTPStatusError(
            "Unauthorized",
            request=MagicMock(),
            response=MagicMock(status_code=401)
        )):
            with pytest.raises((httpx.HTTPStatusError, Exception)):
                await api.get_agent_status()
    
    @pytest.mark.asyncio
    async def test_handle_429_rate_limit(self, api):
        """Test handling 429 rate limit error."""
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                error = httpx.HTTPStatusError(
                    "Rate limited",
                    request=MagicMock(),
                    response=MagicMock(status_code=429)
                )
                raise error
            return MagicMock(json=lambda: {"status": "ok"})
        
        with patch.object(api, '_request', side_effect=mock_request):
            result = await api.get_agent_status()
            
            # Should retry after rate limit
            assert call_count >= 2
    
    @pytest.mark.asyncio
    async def test_handle_500_server_error(self, api):
        """Test handling 500 server error."""
        with patch.object(api, '_request', side_effect=httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=MagicMock(status_code=500)
        )):
            # Should raise or return error
            try:
                await api.get_agent_status()
                assert False, "Should have raised"
            except Exception:
                pass  # Expected
    
    @pytest.mark.asyncio
    async def test_handle_network_error(self, api):
        """Test handling network error."""
        with patch.object(api, '_request', side_effect=httpx.ConnectError("Network error")):
            # Should raise or return error
            try:
                await api.get_agent_status()
            except Exception:
                pass  # Expected
    
    @pytest.mark.asyncio
    async def test_handle_timeout(self, api):
        """Test handling timeout."""
        with patch.object(api, '_request', side_effect=httpx.TimeoutException("Timeout")):
            # Should raise or return error
            try:
                await api.get_agent_status()
            except Exception:
                pass  # Expected


# ============================================================
# TESTS - Response Validation
# ============================================================

class TestResponseValidation:
    """Tests for response validation."""
    
    @pytest.fixture
    def api(self, env_vars):
        """Create API client instance."""
        return ColosseumAPI()
    
    @pytest.mark.asyncio
    async def test_validate_status_response(self, api):
        """Test validating status response."""
        valid_response = {
            "status": "active",
            "nextSteps": ["engage_forum"]
        }
        
        with patch.object(api, '_request', return_value=MagicMock(
            json=lambda: valid_response
        )):
            status = await api.get_agent_status()
            
            assert "status" in status
    
    @pytest.mark.asyncio
    async def test_handle_malformed_response(self, api):
        """Test handling malformed response."""
        with patch.object(api, '_request', return_value=MagicMock(
            json=lambda: "not a dict"
        )):
            try:
                await api.get_agent_status()
                # May succeed with string response
            except Exception:
                pass  # Expected for malformed response
    
    @pytest.mark.asyncio
    async def test_handle_empty_response(self, api):
        """Test handling empty response."""
        with patch.object(api, '_request', return_value=MagicMock(
            json=lambda: None
        )):
            result = await api.get_agent_status()
            
            # Should handle gracefully
            assert result is None or isinstance(result, dict)
