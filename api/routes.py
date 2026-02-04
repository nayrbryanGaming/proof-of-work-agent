"""
API Routes for the POW Agent Dashboard.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from agent.config import config
from agent.logger import get_logger
from api.websocket import manager as ws_manager

router = APIRouter(tags=["agent"])
log = get_logger("api.routes")

# Data paths
LOGS_PATH = Path(__file__).resolve().parent.parent / "logs" / "agent.log"
TASKS_PATH = Path(__file__).resolve().parent.parent / "tasks" / "sample_tasks.json"
STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "state.json"


# ============================================================
# Pydantic Models
# ============================================================

class TaskCreate(BaseModel):
    description: str = Field(..., min_length=10, max_length=1000)
    category: str = Field(default="general", max_length=50)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")


class TaskResponse(BaseModel):
    id: int
    description: str
    category: str
    difficulty: str
    created_at: Optional[str] = None


class AgentState(BaseModel):
    running: bool
    cycle_count: int
    last_cycle_at: Optional[str]
    last_heartbeat_at: Optional[str]
    last_forum_at: Optional[str]
    last_solana_tx: Optional[str]
    errors_count: int
    tasks_solved: int


class CycleResult(BaseModel):
    cycle_number: int
    heartbeat_synced: bool
    status_checked: bool
    forum_engaged: bool
    task_solved: bool
    task_hash: Optional[str]
    solana_tx: Optional[str]
    project_updated: bool
    duration: float
    errors: List[str]
    timestamp: str


class ConfigUpdate(BaseModel):
    loop_interval: Optional[int] = Field(None, ge=60, le=7200)
    log_level: Optional[str] = Field(None, pattern="^(DEBUG|INFO|WARNING|ERROR)$")


class LogEntry(BaseModel):
    timestamp: str
    level: str
    module: str
    message: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_cycles: int
    successful_cycles: int
    failed_cycles: int
    tasks_solved: int
    forum_engagements: int
    solana_transactions: int
    average_cycle_duration: float
    error_rate: float


# ============================================================
# State Management
# ============================================================

_state: Dict[str, Any] = {
    "running": False,
    "cycle_count": 0,
    "last_cycle_at": None,
    "last_heartbeat_at": None,
    "last_forum_at": None,
    "last_solana_tx": None,
    "errors_count": 0,
    "tasks_solved": 0,
    "start_time": None,
    "cycles": [],
    "metrics": {
        "total_cycles": 0,
        "successful_cycles": 0,
        "failed_cycles": 0,
        "tasks_solved": 0,
        "forum_engagements": 0,
        "solana_transactions": 0,
        "cycle_durations": [],
    }
}


def _load_state():
    """Load state from file."""
    global _state
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                saved = json.load(f)
                _state.update(saved)
        except Exception:
            pass


def _save_state():
    """Save state to file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(_state, f, indent=2, default=str)


def update_state(key: str, value: Any):
    """Update state and broadcast via WebSocket."""
    _state[key] = value
    _save_state()
    asyncio.create_task(ws_manager.broadcast({
        "type": "state_update",
        "key": key,
        "value": value,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }))


def record_cycle(result: CycleResult):
    """Record a cycle result."""
    _state["cycle_count"] = result.cycle_number
    _state["last_cycle_at"] = result.timestamp
    _state["metrics"]["total_cycles"] += 1
    
    if result.heartbeat_synced:
        _state["last_heartbeat_at"] = result.timestamp
    
    if result.forum_engaged:
        _state["last_forum_at"] = result.timestamp
        _state["metrics"]["forum_engagements"] += 1
    
    if result.task_solved:
        _state["tasks_solved"] += 1
        _state["metrics"]["tasks_solved"] += 1
    
    if result.solana_tx:
        _state["last_solana_tx"] = result.solana_tx
        _state["metrics"]["solana_transactions"] += 1
    
    if result.errors:
        _state["errors_count"] += len(result.errors)
        _state["metrics"]["failed_cycles"] += 1
    else:
        _state["metrics"]["successful_cycles"] += 1
    
    _state["metrics"]["cycle_durations"].append(result.duration)
    # Keep only last 100 durations
    _state["metrics"]["cycle_durations"] = _state["metrics"]["cycle_durations"][-100:]
    
    _state["cycles"].append(result.dict())
    # Keep only last 50 cycles
    _state["cycles"] = _state["cycles"][-50:]
    
    _save_state()
    
    asyncio.create_task(ws_manager.broadcast({
        "type": "cycle_complete",
        "data": result.dict(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }))


# Load state on module import
_load_state()


# ============================================================
# Routes
# ============================================================

@router.get("/status", response_model=AgentState)
async def get_status():
    """Get current agent status."""
    from api.server import get_agent_status
    
    agent_status = get_agent_status()
    
    return AgentState(
        running=agent_status["running"],
        cycle_count=_state["cycle_count"],
        last_cycle_at=_state["last_cycle_at"],
        last_heartbeat_at=_state["last_heartbeat_at"],
        last_forum_at=_state["last_forum_at"],
        last_solana_tx=_state["last_solana_tx"],
        errors_count=_state["errors_count"],
        tasks_solved=_state["tasks_solved"]
    )


@router.post("/start")
async def start_agent():
    """Start the agent."""
    from api.server import start_agent_background, get_agent_status
    
    status = get_agent_status()
    if status["running"]:
        raise HTTPException(status_code=400, detail="Agent is already running")
    
    _state["start_time"] = datetime.now(timezone.utc).isoformat()
    _state["running"] = True
    _save_state()
    
    success = await start_agent_background()
    
    if success:
        await ws_manager.broadcast({
            "type": "agent_started",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return {"status": "started", "message": "Agent started successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start agent")


@router.post("/stop")
async def stop_agent():
    """Stop the agent."""
    from api.server import stop_agent_background, get_agent_status
    
    status = get_agent_status()
    if not status["running"]:
        raise HTTPException(status_code=400, detail="Agent is not running")
    
    _state["running"] = False
    _save_state()
    
    success = await stop_agent_background()
    
    if success:
        await ws_manager.broadcast({
            "type": "agent_stopped",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return {"status": "stopped", "message": "Agent stopped successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to stop agent")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get agent metrics."""
    metrics = _state["metrics"]
    
    start_time = _state.get("start_time")
    if start_time:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        uptime = (datetime.now(timezone.utc) - start_dt).total_seconds()
    else:
        uptime = 0
    
    durations = metrics.get("cycle_durations", [])
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    total = metrics.get("total_cycles", 0)
    failed = metrics.get("failed_cycles", 0)
    error_rate = (failed / total * 100) if total > 0 else 0
    
    return MetricsResponse(
        uptime_seconds=uptime,
        total_cycles=total,
        successful_cycles=metrics.get("successful_cycles", 0),
        failed_cycles=failed,
        tasks_solved=metrics.get("tasks_solved", 0),
        forum_engagements=metrics.get("forum_engagements", 0),
        solana_transactions=metrics.get("solana_transactions", 0),
        average_cycle_duration=avg_duration,
        error_rate=error_rate
    )


@router.get("/cycles", response_model=List[CycleResult])
async def get_cycles(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Get recent cycle results."""
    cycles = _state.get("cycles", [])
    cycles = sorted(cycles, key=lambda x: x.get("cycle_number", 0), reverse=True)
    return cycles[offset:offset + limit]


@router.get("/logs", response_model=List[LogEntry])
async def get_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    level: Optional[str] = Query(default=None, pattern="^(DEBUG|INFO|WARNING|ERROR)$"),
    module: Optional[str] = Query(default=None)
):
    """Get agent logs."""
    if not LOGS_PATH.exists():
        return []
    
    logs = []
    try:
        with LOGS_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in reversed(lines[-limit * 2:]):
            line = line.strip()
            if not line:
                continue
            
            # Parse log format: [TIME][LEVEL][MODULE] message
            try:
                parts = line.split("]")
                if len(parts) >= 4:
                    timestamp = parts[0][1:]
                    log_level = parts[1][1:]
                    log_module = parts[2][1:]
                    message = "]".join(parts[3:]).strip()
                    
                    if level and log_level != level:
                        continue
                    if module and module not in log_module:
                        continue
                    
                    logs.append(LogEntry(
                        timestamp=timestamp,
                        level=log_level,
                        module=log_module,
                        message=message
                    ))
                    
                    if len(logs) >= limit:
                        break
            except Exception:
                continue
    except Exception as e:
        log.error(f"Failed to read logs: {e}")
    
    return logs


@router.get("/tasks", response_model=List[TaskResponse])
async def get_tasks():
    """Get all tasks."""
    if not TASKS_PATH.exists():
        return []
    
    try:
        with TASKS_PATH.open("r", encoding="utf-8") as f:
            tasks = json.load(f)
        return tasks
    except Exception as e:
        log.error(f"Failed to load tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to load tasks")


@router.post("/tasks", response_model=TaskResponse)
async def create_task(task: TaskCreate):
    """Create a new task."""
    tasks = []
    if TASKS_PATH.exists():
        try:
            with TASKS_PATH.open("r", encoding="utf-8") as f:
                tasks = json.load(f)
        except Exception:
            pass
    
    new_id = max([t.get("id", 0) for t in tasks], default=0) + 1
    new_task = {
        "id": new_id,
        "description": task.description,
        "category": task.category,
        "difficulty": task.difficulty,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    tasks.append(new_task)
    
    with TASKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4)
    
    await ws_manager.broadcast({
        "type": "task_created",
        "data": new_task,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    return TaskResponse(**new_task)


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task."""
    if not TASKS_PATH.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        with TASKS_PATH.open("r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load tasks")
    
    original_len = len(tasks)
    tasks = [t for t in tasks if t.get("id") != task_id]
    
    if len(tasks) == original_len:
        raise HTTPException(status_code=404, detail="Task not found")
    
    with TASKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4)
    
    return {"status": "deleted", "task_id": task_id}


@router.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)."""
    return {
        "colosseum_base_url": config.colosseum.base_url if hasattr(config, 'colosseum') else config.colosseum_base_url,
        "solana_rpc": config.solana.rpc_url if hasattr(config, 'solana') else config.solana_rpc,
        "program_id": (config.solana.program_id if hasattr(config, 'solana') else config.program_id) or "NOT_SET",
        "loop_interval": config.agent.loop_interval if hasattr(config, 'agent') else 1800,
        "log_level": config.agent.log_level if hasattr(config, 'agent') else "INFO",
        "heartbeat_url": config.agent.heartbeat_url if hasattr(config, 'agent') else "https://colosseum.com/heartbeat.md"
    }


@router.post("/trigger-cycle")
async def trigger_cycle():
    """Manually trigger a single cycle."""
    from api.server import get_agent_status
    
    status = get_agent_status()
    if status["running"]:
        raise HTTPException(
            status_code=400, 
            detail="Cannot trigger manual cycle while agent is running"
        )
    
    try:
        from agent.loop import run_cycle
        from colosseum.api import ColosseumAPI
        from solana.client import SolanaClient
        
        api = ColosseumAPI()
        solana = SolanaClient()
        
        result = await run_cycle(api, solana, _state["cycle_count"] + 1)
        
        cycle_result = CycleResult(
            cycle_number=result.cycle_number,
            heartbeat_synced=result.heartbeat_synced,
            status_checked=result.status_checked,
            forum_engaged=result.forum_engaged,
            task_solved=result.task_solved,
            task_hash=result.task_hash,
            solana_tx=result.solana_tx,
            project_updated=result.project_updated,
            duration=result.duration,
            errors=result.errors,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        record_cycle(cycle_result)
        
        return {"status": "completed", "result": cycle_result.dict()}
    
    except Exception as e:
        log.error(f"Manual cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# WebSocket
# ============================================================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)
    
    # Send initial state
    await websocket.send_json({
        "type": "initial_state",
        "data": {
            "running": _state["running"],
            "cycle_count": _state["cycle_count"],
            "last_cycle_at": _state["last_cycle_at"],
            "metrics": _state["metrics"]
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle client messages
            if data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)
