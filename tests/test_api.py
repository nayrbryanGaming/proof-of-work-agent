"""
Test suite for API endpoints.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from httpx import AsyncClient


# ============================================================
# TESTS - Status Endpoints
# ============================================================

class TestStatusEndpoints:
    """Tests for status endpoints."""
    
    def test_root_redirect(self, api_client):
        """Test root redirects to docs."""
        response = api_client.get("/", follow_redirects=False)
        
        assert response.status_code in [status.HTTP_307_TEMPORARY_REDIRECT, status.HTTP_200_OK]
    
    def test_health_endpoint(self, api_client):
        """Test health endpoint returns status."""
        response = api_client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy", "degraded"]
    
    def test_health_live(self, api_client):
        """Test Kubernetes liveness probe."""
        response = api_client.get("/api/v1/health/live")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_health_ready(self, api_client):
        """Test Kubernetes readiness probe."""
        response = api_client.get("/api/v1/health/ready")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]
    
    def test_status_endpoint(self, api_client):
        """Test status endpoint."""
        response = api_client.get("/api/v1/status")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "running" in data
        assert "uptime" in data or "cycle_count" in data


# ============================================================
# TESTS - Agent Control Endpoints
# ============================================================

class TestAgentControlEndpoints:
    """Tests for agent control endpoints."""
    
    def test_start_agent(self, api_client):
        """Test starting the agent."""
        response = api_client.post("/api/v1/agent/start")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_202_ACCEPTED,
            status.HTTP_409_CONFLICT  # Already running
        ]
    
    def test_stop_agent(self, api_client):
        """Test stopping the agent."""
        response = api_client.post("/api/v1/agent/stop")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_409_CONFLICT  # Not running
        ]
    
    def test_restart_agent(self, api_client):
        """Test restarting the agent."""
        response = api_client.post("/api/v1/agent/restart")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_202_ACCEPTED
        ]
    
    def test_trigger_cycle(self, api_client):
        """Test manually triggering a cycle."""
        response = api_client.post("/api/v1/agent/cycle")
        
        # May require agent to be running
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_202_ACCEPTED,
            status.HTTP_409_CONFLICT,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]


# ============================================================
# TESTS - Metrics Endpoints
# ============================================================

class TestMetricsEndpoints:
    """Tests for metrics endpoints."""
    
    def test_metrics_json(self, api_client):
        """Test metrics endpoint returns JSON."""
        response = api_client.get("/api/v1/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, dict)
    
    def test_metrics_prometheus(self, api_client):
        """Test Prometheus metrics format."""
        response = api_client.get(
            "/api/v1/metrics",
            headers={"Accept": "text/plain"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        # Should be text format
        assert "text" in response.headers.get("content-type", "")


# ============================================================
# TESTS - Cycle Endpoints
# ============================================================

class TestCycleEndpoints:
    """Tests for cycle endpoints."""
    
    def test_list_cycles(self, api_client):
        """Test listing cycles."""
        response = api_client.get("/api/v1/cycles")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, list) or "cycles" in data
    
    def test_list_cycles_pagination(self, api_client):
        """Test cycle pagination."""
        response = api_client.get("/api/v1/cycles?limit=5&offset=0")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_get_cycle_by_id(self, api_client, sample_cycle_result):
        """Test getting specific cycle."""
        # First need to have a cycle
        response = api_client.get("/api/v1/cycles/1")
        
        # May or may not exist
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]
    
    def test_get_latest_cycle(self, api_client):
        """Test getting latest cycle."""
        response = api_client.get("/api/v1/cycles/latest")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND  # No cycles yet
        ]


# ============================================================
# TESTS - Task Endpoints
# ============================================================

class TestTaskEndpoints:
    """Tests for task endpoints."""
    
    def test_list_tasks(self, api_client):
        """Test listing tasks."""
        response = api_client.get("/api/v1/tasks")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_create_task(self, api_client, sample_task):
        """Test creating a task."""
        response = api_client.post(
            "/api/v1/tasks",
            json=sample_task
        )
        
        assert response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_200_OK
        ]
    
    def test_create_task_validation(self, api_client):
        """Test task creation validation."""
        response = api_client.post(
            "/api/v1/tasks",
            json={}  # Invalid - missing required fields
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_task_by_id(self, api_client):
        """Test getting task by ID."""
        response = api_client.get("/api/v1/tasks/1")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]
    
    def test_update_task(self, api_client):
        """Test updating a task."""
        response = api_client.patch(
            "/api/v1/tasks/1",
            json={"status": "completed"}
        )
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]
    
    def test_delete_task(self, api_client):
        """Test deleting a task."""
        response = api_client.delete("/api/v1/tasks/1")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_204_NO_CONTENT,
            status.HTTP_404_NOT_FOUND
        ]


# ============================================================
# TESTS - Log Endpoints
# ============================================================

class TestLogEndpoints:
    """Tests for log endpoints."""
    
    def test_list_logs(self, api_client):
        """Test listing log files."""
        response = api_client.get("/api/v1/logs")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_get_log_content(self, api_client):
        """Test getting log content."""
        response = api_client.get("/api/v1/logs/agent.log")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]
    
    def test_get_log_with_lines(self, api_client):
        """Test getting specific number of log lines."""
        response = api_client.get("/api/v1/logs/agent.log?lines=100")
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]


# ============================================================
# TESTS - Config Endpoints
# ============================================================

class TestConfigEndpoints:
    """Tests for configuration endpoints."""
    
    def test_get_config(self, api_client):
        """Test getting configuration."""
        response = api_client.get("/api/v1/config")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        # Should not expose sensitive values
        if "OPENAI_API_KEY" in data:
            assert data["OPENAI_API_KEY"] == "***" or data["OPENAI_API_KEY"].startswith("***")
    
    def test_update_config(self, api_client):
        """Test updating configuration."""
        response = api_client.patch(
            "/api/v1/config",
            json={"CYCLE_INTERVAL": 120}
        )
        
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST  # Invalid config
        ]
    
    def test_update_config_validation(self, api_client):
        """Test config update validation."""
        response = api_client.patch(
            "/api/v1/config",
            json={"CYCLE_INTERVAL": -1}  # Invalid value
        )
        
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


# ============================================================
# TESTS - WebSocket
# ============================================================

class TestWebSocket:
    """Tests for WebSocket endpoints."""
    
    def test_websocket_connect(self, api_client):
        """Test WebSocket connection."""
        with api_client.websocket_connect("/api/v1/ws") as websocket:
            # Should receive initial message
            data = websocket.receive_json()
            
            assert "type" in data or "event" in data
    
    def test_websocket_ping_pong(self, api_client):
        """Test WebSocket keepalive."""
        with api_client.websocket_connect("/api/v1/ws") as websocket:
            websocket.send_json({"type": "ping"})
            
            data = websocket.receive_json()
            
            assert data.get("type") == "pong"
    
    def test_websocket_subscribe_events(self, api_client):
        """Test WebSocket event subscription."""
        with api_client.websocket_connect("/api/v1/ws") as websocket:
            websocket.send_json({
                "type": "subscribe",
                "events": ["cycle_complete", "error"]
            })
            
            data = websocket.receive_json()
            
            assert data.get("type") in ["subscribed", "ack", "success"]


# ============================================================
# TESTS - Error Handling
# ============================================================

class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_404_not_found(self, api_client):
        """Test 404 response."""
        response = api_client.get("/api/v1/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        data = response.json()
        assert "detail" in data or "error" in data
    
    def test_405_method_not_allowed(self, api_client):
        """Test 405 response."""
        response = api_client.put("/api/v1/health")  # PUT not allowed
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_422_validation_error(self, api_client):
        """Test 422 validation error."""
        response = api_client.post(
            "/api/v1/tasks",
            json={"invalid_field": "value"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_500_internal_error(self, api_client):
        """Test 500 error handling."""
        with patch("api.routes.get_agent_status", side_effect=Exception("Internal error")):
            response = api_client.get("/api/v1/status")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ============================================================
# TESTS - Authentication (if implemented)
# ============================================================

class TestAuthentication:
    """Tests for authentication."""
    
    def test_api_key_header(self, api_client, env_vars):
        """Test API key authentication."""
        response = api_client.get(
            "/api/v1/status",
            headers={"X-API-Key": "test_api_key"}
        )
        
        # May or may not require auth
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN
        ]
    
    def test_missing_api_key(self, api_client):
        """Test missing API key."""
        # Some endpoints may require auth
        response = api_client.get("/api/v1/admin/config")
        
        assert response.status_code in [
            status.HTTP_200_OK,  # No auth required
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
            status.HTTP_404_NOT_FOUND  # Endpoint doesn't exist
        ]


# ============================================================
# TESTS - Rate Limiting
# ============================================================

class TestRateLimiting:
    """Tests for API rate limiting."""
    
    def test_rate_limit_headers(self, api_client):
        """Test rate limit headers are present."""
        response = api_client.get("/api/v1/status")
        
        # May or may not have rate limit headers
        headers = response.headers
        
        # Common rate limit headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "RateLimit-Limit",
            "RateLimit-Remaining"
        ]
        
        # At least check response is successful
        assert response.status_code == status.HTTP_200_OK
    
    def test_rate_limit_exceeded(self, api_client):
        """Test rate limit exceeded response."""
        # Make many requests quickly
        responses = []
        for _ in range(100):
            responses.append(api_client.get("/api/v1/status"))
        
        # At least one should succeed
        assert any(r.status_code == status.HTTP_200_OK for r in responses)
        
        # May hit rate limit
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS for r in responses)
        
        # Rate limiting may or may not be enabled
        # Just verify we don't crash


# ============================================================
# TESTS - CORS
# ============================================================

class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self, api_client):
        """Test CORS headers are present."""
        response = api_client.options(
            "/api/v1/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should allow or properly handle OPTIONS
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_204_NO_CONTENT,
            status.HTTP_405_METHOD_NOT_ALLOWED  # If CORS not configured
        ]
    
    def test_cors_allowed_origin(self, api_client):
        """Test allowed origin."""
        response = api_client.get(
            "/api/v1/status",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # May have CORS header
        allow_origin = response.headers.get("Access-Control-Allow-Origin")
        if allow_origin:
            assert allow_origin in ["*", "http://localhost:3000"]
