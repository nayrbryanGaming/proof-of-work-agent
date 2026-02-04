"""
FastAPI Backend for Proof-of-Work Agent Dashboard.
Provides REST API and WebSocket connections for real-time monitoring.
"""

from .server import app, create_app
from .routes import router
from .websocket import ConnectionManager

__all__ = ["app", "create_app", "router", "ConnectionManager"]
