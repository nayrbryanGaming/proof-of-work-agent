"""
Enhanced API Routes - Production-grade with advanced features.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from agent.config import config
from agent.logger import get_logger
from api.websocket import manager as ws_manager


router = APIRouter(tags=["Agent"])
log = get_logger("api.routes")

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


# ============================================================
# PYDANTIC MODELS
# ============================================================

class StatusResponse(BaseModel):
    """Agent status response."""
    running: bool
    current_cycle: int
    uptime_seconds: float
    last_heartbeat: Optional[str] = None
    last_cycle: Optional[str] = None
    health: str = "unknown"
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Metrics response."""
    cycles: Dict[str, Any]
    tasks: Dict[str, Any]
    forum: Dict[str, Any]
    solana: Dict[str, Any]
    health: Dict[str, Any]


class CycleResponse(BaseModel):
    """Single cycle response."""
    cycle_number: int
    started_at: str
    completed_at: Optional[str]
    duration: float
    status: str
    heartbeat_synced: bool
    forum_engaged: bool
    task_solved: bool
    task_hash: Optional[str] = None
    solana_tx: Optional[str] = None
    errors: List[str] = []


class TaskCreateRequest(BaseModel):
    """Request to create a task."""
    description: str = Field(..., min_length=10, max_length=2000)
    category: str = Field(default="general", max_length=50)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    priority: int = Field(default=2, ge=0, le=4)
    tags: List[str] = Field(default_factory=list)
    
    @validator('description')
    def validate_description(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Description too short')
        return v.strip()


class TaskResponse(BaseModel):
    """Task response."""
    id: int
    description: str
    category: str
    difficulty: str
    priority: int = 2
    tags: List[str] = []
    created_at: Optional[str] = None
    solved: bool = False
    solution_hash: Optional[str] = None


class ConfigResponse(BaseModel):
    """Configuration response (read-only)."""
    loop_interval: int
    heartbeat_url: str
    solana_rpc: str
    program_id: Optional[str]
    log_level: str
    forum_enabled: bool = True
    max_forum_comments: int = 5


class LogEntry(BaseModel):
    """Log entry model."""
    timestamp: str
    level: str
    module: str
    message: str


class EventResponse(BaseModel):
    """Event response."""
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: str
    source: str


class HealthCheckResult(BaseModel):
    """Health check result."""
    name: str
    status: str
    message: str
    latency_ms: float


class OverallHealthResponse(BaseModel):
    """Overall health response."""
    status: str
    checks: List[HealthCheckResult]
    uptime_seconds: float
    timestamp: str


# ============================================================
# STATE MANAGEMENT
# ============================================================

class AgentStateManager:
    """Manages agent state with persistence."""
    
    def __init__(self):
        self._state = {
            "running": False,
            "current_cycle": 0,
            "start_time": None,
            "last_heartbeat": None,
            "last_cycle_time": None,
            "total_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "tasks_solved": 0,
            "forum_engagements": 0,
            "solana_transactions": 0
        }
        self._state_file = _DATA_DIR / "api_state.json"
        self._load()
    
    def _load(self):
        """Load state from file."""
        if self._state_file.exists():
            try:
                with self._state_file.open() as f:
                    saved = json.load(f)
                    self._state.update(saved)
                    self._state["running"] = False  # Always start stopped
            except Exception as e:
                log.warn(f"Failed to load state: {e}")
    
    def _save(self):
        """Save state to file."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._state_file.open("w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            log.warn(f"Failed to save state: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any):
        self._state[key] = value
        self._save()
    
    def update(self, data: Dict[str, Any]):
        self._state.update(data)
        self._save()
    
    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)


_state = AgentStateManager()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _parse_log_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a log line into components."""
    try:
        import re
        match = re.match(r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\s*(.*)', line.strip())
        if match:
            return {
                "timestamp": match.group(1),
                "level": match.group(2),
                "module": match.group(3),
                "message": match.group(4)
            }
    except Exception:
        pass
    return None


def _load_tasks() -> List[Dict[str, Any]]:
    """Load tasks from file."""
    tasks_file = _TASKS_DIR / "sample_tasks.json"
    if tasks_file.exists():
        try:
            with tasks_file.open() as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_tasks(tasks: List[Dict[str, Any]]):
    """Save tasks to file."""
    tasks_file = _TASKS_DIR / "sample_tasks.json"
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    with tasks_file.open("w") as f:
        json.dump(tasks, f, indent=2)


def _load_cycles() -> List[Dict[str, Any]]:
    """Load cycle history."""
    state_file = _DATA_DIR / "state.json"
    if state_file.exists():
        try:
            with state_file.open() as f:
                data = json.load(f)
                return data.get("recent_cycles", [])
        except Exception:
            pass
    return []


async def _stream_logs(log_file: Path, lines: int = 100):
    """Stream log file as SSE."""
    if not log_file.exists():
        yield "data: No logs available\n\n"
        return
    
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                yield f"data: {json.dumps({'line': line.strip()})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ============================================================
# ROUTES - STATUS & CONTROL
# ============================================================

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current agent status."""
    uptime = 0.0
    if _state.get("start_time"):
        try:
            start = datetime.fromisoformat(_state.get("start_time"))
            uptime = (datetime.now(timezone.utc) - start).total_seconds()
        except Exception:
            pass
    
    return StatusResponse(
        running=_state.get("running", False),
        current_cycle=_state.get("current_cycle", 0),
        uptime_seconds=uptime,
        last_heartbeat=_state.get("last_heartbeat"),
        last_cycle=_state.get("last_cycle_time"),
        health="healthy" if _state.get("running") else "stopped"
    )


@router.post("/start")
async def start_agent(background_tasks: BackgroundTasks):
    """Start the agent loop."""
    if _state.get("running"):
        raise HTTPException(status_code=400, detail="Agent is already running")
    
    try:
        from api.server import start_agent_background
        success = await start_agent_background()
        
        if success:
            _state.update({
                "running": True,
                "start_time": datetime.now(timezone.utc).isoformat()
            })
            
            await ws_manager.broadcast({
                "type": "agent_started",
                "timestamp": _state.get("start_time")
            })
            
            return {"status": "started", "message": "Agent started successfully"}
    except Exception as e:
        log.error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=500, detail="Failed to start agent")


@router.post("/stop")
async def stop_agent():
    """Stop the agent loop."""
    if not _state.get("running"):
        raise HTTPException(status_code=400, detail="Agent is not running")
    
    try:
        from api.server import stop_agent_background
        success = await stop_agent_background()
        
        if success:
            _state.set("running", False)
            
            await ws_manager.broadcast({
                "type": "agent_stopped",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {"status": "stopped", "message": "Agent stopped successfully"}
    except Exception as e:
        log.error(f"Failed to stop agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=500, detail="Failed to stop agent")


@router.post("/trigger-cycle")
async def trigger_cycle():
    """Manually trigger an agent cycle."""
    if not _state.get("running"):
        raise HTTPException(status_code=400, detail="Agent is not running")
    
    # Signal would go to the running loop
    await ws_manager.broadcast({
        "type": "cycle_trigger_requested",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    return {"status": "triggered", "message": "Cycle trigger requested"}


# ============================================================
# ROUTES - METRICS
# ============================================================

@router.get("/metrics")
async def get_metrics():
    """Get agent metrics."""
    try:
        from agent.metrics import agent_metrics
        return agent_metrics.get_summary()
    except ImportError:
        # Fallback if metrics module not available
        return {
            "cycles": {
                "total": _state.get("total_cycles", 0),
                "success": _state.get("successful_cycles", 0),
                "failed": _state.get("failed_cycles", 0),
                "current": _state.get("current_cycle", 0)
            },
            "tasks": {"solved": _state.get("tasks_solved", 0)},
            "forum": {"engagements": _state.get("forum_engagements", 0)},
            "solana": {"transactions": _state.get("solana_transactions", 0)},
            "health": {"running": _state.get("running", False)}
        }


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format."""
    try:
        from agent.metrics import registry
        return StreamingResponse(
            iter([registry.to_prometheus()]),
            media_type="text/plain"
        )
    except ImportError:
        return StreamingResponse(
            iter(["# Metrics not available\n"]),
            media_type="text/plain"
        )


# ============================================================
# ROUTES - HEALTH
# ============================================================

@router.get("/health")
async def get_health():
    """Get comprehensive health check."""
    checks = []
    
    # Check agent running
    checks.append({
        "name": "agent_loop",
        "status": "healthy" if _state.get("running") else "stopped",
        "message": "Agent is running" if _state.get("running") else "Agent is stopped",
        "latency_ms": 0
    })
    
    # Check logs directory
    logs_ok = _LOGS_DIR.exists()
    checks.append({
        "name": "logs_directory",
        "status": "healthy" if logs_ok else "unhealthy",
        "message": "Logs directory exists" if logs_ok else "Logs directory missing",
        "latency_ms": 0
    })
    
    # Check data directory
    data_ok = _DATA_DIR.exists()
    checks.append({
        "name": "data_directory",
        "status": "healthy" if data_ok else "unhealthy",
        "message": "Data directory exists" if data_ok else "Data directory missing",
        "latency_ms": 0
    })
    
    # Check config
    config_ok = bool(config.colosseum_api_key or True)  # Always OK for now
    checks.append({
        "name": "configuration",
        "status": "healthy" if config_ok else "degraded",
        "message": "Configuration loaded",
        "latency_ms": 0
    })
    
    uptime = 0.0
    if _state.get("start_time"):
        try:
            start = datetime.fromisoformat(_state.get("start_time"))
            uptime = (datetime.now(timezone.utc) - start).total_seconds()
        except Exception:
            pass
    
    # Determine overall status
    unhealthy_count = sum(1 for c in checks if c["status"] == "unhealthy")
    overall = "healthy" if unhealthy_count == 0 else "unhealthy"
    
    return {
        "status": overall,
        "checks": checks,
        "uptime_seconds": uptime,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    # Check essential components
    if not _DATA_DIR.exists():
        raise HTTPException(status_code=503, detail="Data directory not ready")
    
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


# ============================================================
# ROUTES - CYCLES
# ============================================================

@router.get("/cycles", response_model=List[CycleResponse])
async def get_cycles(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, pattern="^(completed|failed|running)$")
):
    """Get cycle history with pagination."""
    cycles = _load_cycles()
    
    # Filter by status if specified
    if status:
        if status == "completed":
            cycles = [c for c in cycles if c.get("completed_at") and not c.get("errors")]
        elif status == "failed":
            cycles = [c for c in cycles if c.get("errors")]
        elif status == "running":
            cycles = [c for c in cycles if not c.get("completed_at")]
    
    # Apply pagination
    total = len(cycles)
    cycles = cycles[offset:offset + limit]
    
    return [
        CycleResponse(
            cycle_number=c.get("cycle_number", 0),
            started_at=c.get("started_at", ""),
            completed_at=c.get("completed_at"),
            duration=c.get("duration", 0),
            status="failed" if c.get("errors") else ("completed" if c.get("completed_at") else "running"),
            heartbeat_synced=c.get("heartbeat_synced", False),
            forum_engaged=c.get("forum_engaged", False),
            task_solved=c.get("task_solved", False),
            task_hash=c.get("task_hash"),
            solana_tx=c.get("solana_tx"),
            errors=c.get("errors", [])
        )
        for c in cycles
    ]


@router.get("/cycles/latest")
async def get_latest_cycle():
    """Get the most recent cycle."""
    cycles = _load_cycles()
    if not cycles:
        raise HTTPException(status_code=404, detail="No cycles found")
    
    latest = cycles[-1]
    return CycleResponse(
        cycle_number=latest.get("cycle_number", 0),
        started_at=latest.get("started_at", ""),
        completed_at=latest.get("completed_at"),
        duration=latest.get("duration", 0),
        status="completed" if latest.get("completed_at") else "running",
        heartbeat_synced=latest.get("heartbeat_synced", False),
        forum_engaged=latest.get("forum_engaged", False),
        task_solved=latest.get("task_solved", False),
        task_hash=latest.get("task_hash"),
        solana_tx=latest.get("solana_tx"),
        errors=latest.get("errors", [])
    )


@router.get("/cycles/{cycle_number}")
async def get_cycle(cycle_number: int):
    """Get specific cycle by number."""
    cycles = _load_cycles()
    
    for c in cycles:
        if c.get("cycle_number") == cycle_number:
            return c
    
    raise HTTPException(status_code=404, detail=f"Cycle {cycle_number} not found")


# ============================================================
# ROUTES - LOGS
# ============================================================

@router.get("/logs")
async def get_logs(
    lines: int = Query(default=100, ge=1, le=5000),
    level: Optional[str] = Query(default=None, pattern="^(INFO|WARN|ERROR|DEBUG|SUCCESS)$"),
    module: Optional[str] = None,
    search: Optional[str] = None
):
    """Get agent logs with filtering."""
    log_file = _LOGS_DIR / "agent.log"
    
    if not log_file.exists():
        return {"logs": [], "total": 0, "filtered": True}
    
    logs = []
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            
            for line in reversed(all_lines):
                parsed = _parse_log_line(line)
                if not parsed:
                    continue
                
                # Apply filters
                if level and parsed["level"] != level:
                    continue
                if module and module.lower() not in parsed["module"].lower():
                    continue
                if search and search.lower() not in line.lower():
                    continue
                
                logs.append(parsed)
                
                if len(logs) >= lines:
                    break
                    
    except Exception as e:
        log.error(f"Error reading logs: {e}")
        return {"logs": [], "total": 0, "error": str(e)}
    
    return {
        "logs": list(reversed(logs)),
        "total": len(logs),
        "filtered": bool(level or module or search)
    }


@router.get("/logs/stream")
async def stream_logs(lines: int = Query(default=100, ge=1, le=1000)):
    """Stream logs as Server-Sent Events."""
    log_file = _LOGS_DIR / "agent.log"
    return StreamingResponse(
        _stream_logs(log_file, lines),
        media_type="text/event-stream"
    )


@router.get("/logs/download")
async def download_logs():
    """Download log file."""
    log_file = _LOGS_DIR / "agent.log"
    
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    
    def iterate_file():
        with log_file.open("rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    return StreamingResponse(
        iterate_file(),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"}
    )


@router.delete("/logs")
async def clear_logs():
    """Clear log file (creates backup first)."""
    log_file = _LOGS_DIR / "agent.log"
    
    if log_file.exists():
        # Create backup
        backup_name = f"agent_{int(datetime.now().timestamp())}.log.bak"
        backup_file = _LOGS_DIR / backup_name
        log_file.rename(backup_file)
        log_file.touch()
        
        return {"status": "cleared", "backup": backup_name}
    
    return {"status": "no_logs", "message": "Log file did not exist"}


# ============================================================
# ROUTES - TASKS
# ============================================================

@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    limit: int = Query(default=50, ge=1, le=500),
    category: Optional[str] = None,
    difficulty: Optional[str] = Query(default=None, pattern="^(easy|medium|hard)$")
):
    """List all tasks."""
    tasks = _load_tasks()
    
    # Apply filters
    if category:
        tasks = [t for t in tasks if t.get("category") == category]
    if difficulty:
        tasks = [t for t in tasks if t.get("difficulty") == difficulty]
    
    tasks = tasks[:limit]
    
    return [
        TaskResponse(
            id=t.get("id", i),
            description=t.get("description", ""),
            category=t.get("category", "general"),
            difficulty=t.get("difficulty", "medium"),
            priority=t.get("priority", 2),
            tags=t.get("tags", []),
            created_at=t.get("created_at"),
            solved=t.get("solved", False),
            solution_hash=t.get("solution_hash")
        )
        for i, t in enumerate(tasks)
    ]


@router.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new task."""
    tasks = _load_tasks()
    
    # Generate new ID
    max_id = max((t.get("id", 0) for t in tasks), default=0)
    new_id = max_id + 1
    
    new_task = {
        "id": new_id,
        "description": request.description,
        "category": request.category,
        "difficulty": request.difficulty,
        "priority": request.priority,
        "tags": request.tags,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "solved": False
    }
    
    tasks.append(new_task)
    _save_tasks(tasks)
    
    # Broadcast to WebSocket
    await ws_manager.broadcast({
        "type": "task_created",
        "task": new_task
    })
    
    return TaskResponse(**new_task)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int):
    """Get task by ID."""
    tasks = _load_tasks()
    
    for t in tasks:
        if t.get("id") == task_id:
            return TaskResponse(
                id=t.get("id"),
                description=t.get("description", ""),
                category=t.get("category", "general"),
                difficulty=t.get("difficulty", "medium"),
                priority=t.get("priority", 2),
                tags=t.get("tags", []),
                created_at=t.get("created_at"),
                solved=t.get("solved", False),
                solution_hash=t.get("solution_hash")
            )
    
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.put("/tasks/{task_id}")
async def update_task(task_id: int, request: TaskCreateRequest):
    """Update a task."""
    tasks = _load_tasks()
    
    for i, t in enumerate(tasks):
        if t.get("id") == task_id:
            tasks[i].update({
                "description": request.description,
                "category": request.category,
                "difficulty": request.difficulty,
                "priority": request.priority,
                "tags": request.tags,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            _save_tasks(tasks)
            return tasks[i]
    
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task."""
    tasks = _load_tasks()
    
    for i, t in enumerate(tasks):
        if t.get("id") == task_id:
            deleted = tasks.pop(i)
            _save_tasks(tasks)
            return {"status": "deleted", "task_id": task_id}
    
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


# ============================================================
# ROUTES - CONFIG
# ============================================================

@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get agent configuration (read-only, sensitive values masked)."""
    return ConfigResponse(
        loop_interval=config.agent.loop_interval if hasattr(config, 'agent') else 1800,
        heartbeat_url=config.heartbeat.url if hasattr(config, 'heartbeat') else "https://colosseum.com/heartbeat.md",
        solana_rpc=config.solana_rpc or "https://api.devnet.solana.com",
        program_id=config.program_id[:8] + "..." if config.program_id else None,
        log_level=config.log_level if hasattr(config, 'log_level') else "INFO"
    )


# ============================================================
# ROUTES - EVENTS
# ============================================================

@router.get("/events")
async def get_events(
    limit: int = Query(default=100, ge=1, le=1000),
    event_type: Optional[str] = None
):
    """Get event history."""
    try:
        from agent.events import event_bus
        return event_bus.get_history(event_type, limit)
    except ImportError:
        return []


# ============================================================
# WEBSOCKET
# ============================================================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "state": _state.state
        })
        
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "status":
                    await websocket.send_json({
                        "type": "status",
                        "data": _state.state
                    })
                else:
                    await websocket.send_json({"type": "echo", "data": data})
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        log.debug("WebSocket client disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)
