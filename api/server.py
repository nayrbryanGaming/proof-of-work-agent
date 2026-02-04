"""
FastAPI Server - Production-grade API backend.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import config
from agent.logger import get_logger
from api.routes import router
from api.websocket import manager as ws_manager


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = {}
        self.window = 60.0
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now(timezone.utc).timestamp()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Clean old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] 
            if now - t < self.window
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."}
            )
        
        self.requests[client_ip].append(now)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests."""
    
    def __init__(self, app):
        super().__init__(app)
        self.log = get_logger("api.request")
    
    async def dispatch(self, request: Request, call_next):
        start = datetime.now(timezone.utc)
        
        response = await call_next(request)
        
        duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        self.log.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration:.2f}ms"
        )
        
        return response


# Global state
_agent_task: Optional[asyncio.Task] = None
_agent_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    log = get_logger("api.server")
    log.info("Starting API server...")
    
    # Startup
    yield
    
    # Shutdown
    global _agent_task, _agent_running
    if _agent_task and not _agent_task.done():
        _agent_running = False
        _agent_task.cancel()
        try:
            await _agent_task
        except asyncio.CancelledError:
            pass
    
    log.info("API server shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Proof-of-Work Agent API",
        description="REST API and WebSocket for POW Agent Dashboard",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    
    # CORS for frontend
    origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", ""),
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware, requests_per_minute=120)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Include routes
    app.include_router(router, prefix="/api")
    
    @app.get("/")
    async def root():
        return {
            "name": "Proof-of-Work Agent API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/api/docs"
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_running": _agent_running
        }
    
    return app


app = create_app()


async def start_agent_background():
    """Start the agent in background."""
    global _agent_task, _agent_running
    
    if _agent_running:
        return False
    
    from agent.loop import forever
    
    _agent_running = True
    _agent_task = asyncio.create_task(forever())
    return True


async def stop_agent_background():
    """Stop the background agent."""
    global _agent_task, _agent_running
    
    if not _agent_running:
        return False
    
    _agent_running = False
    if _agent_task:
        _agent_task.cancel()
        try:
            await _agent_task
        except asyncio.CancelledError:
            pass
    
    return True


def get_agent_status() -> dict:
    """Get current agent status."""
    return {
        "running": _agent_running,
        "task_active": _agent_task is not None and not _agent_task.done() if _agent_task else False
    }
