"""
Main async loop for the Proof-of-Work Agent.
Implements the observe → think → act → verify cycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from agent.logger import get_logger
from agent.config import config
from agent import heartbeat
from agent.heartbeat import HeartbeatChecker
from agent.decision import DecisionEngine, solve, forum_comment
from colosseum.api import ColosseumAPI
from colosseum.forum import run as run_forum, ForumHandler
from colosseum.project import ensure_project, build_update_text, ProjectManager
from colosseum.status import should_act, StatusChecker

# Lazy import SolanaClient to allow running without PROGRAM_ID
_solana_client = None

def get_solana_client():
    """Get Solana client, or None if not configured."""
    global _solana_client
    if _solana_client is not None:
        return _solana_client
    
    program_id = os.getenv("PROGRAM_ID", "").strip()
    if not program_id:
        return None
    
    try:
        from solana.client import SolanaClient
        _solana_client = SolanaClient()
        return _solana_client
    except Exception as e:
        get_logger("loop").warn(f"Solana client not available: {e}")
        return None

_TASKS_PATH = Path(__file__).resolve().parents[1] / "tasks" / "sample_tasks.json"


@dataclass
class CycleResult:
    """Result of a single agent cycle."""
    cycle_number: int
    heartbeat_synced: bool
    status_checked: bool
    forum_engaged: bool
    task_solved: bool
    task_hash: Optional[str]
    solana_tx: Optional[str]
    project_updated: bool
    duration: float
    errors: list[str]


def _load_tasks() -> List[Dict]:
    if not _TASKS_PATH.exists():
        return []
    with _TASKS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_task(tasks: List[Dict], cycle: int = 0) -> Optional[Dict]:
    if not tasks:
        return None
    return tasks[cycle % len(tasks)]


async def run_cycle(api: ColosseumAPI, solana, cycle_num: int = 0) -> CycleResult:
    """Run a single observe → think → act → verify cycle.
    
    Args:
        api: Colosseum API client
        solana: Solana client or None if not configured
        cycle_num: Current cycle number
    """
    log = get_logger("loop")
    start_time = time.time()
    errors: list[str] = []
    
    heartbeat_synced = False
    status_checked = False
    forum_engaged = False
    task_solved = False
    task_hash = None
    solana_tx = None
    project_updated = False
    
    # OBSERVE: heartbeat
    log.info("OBSERVE: heartbeat")
    try:
        heartbeat.check()
        heartbeat_synced = True
    except Exception as e:
        log.error(f"Heartbeat failed: {e}")
        errors.append(str(e))

    # OBSERVE: agent status
    log.info("OBSERVE: agent status")
    try:
        status = api.get_agent_status()
        status_checked = True
        next_steps = status.get("nextSteps") if isinstance(status, dict) else None
        if next_steps:
            log.info(f"Next steps: {next_steps}")
        if not should_act(status):
            log.warn("Status indicates no action for this cycle.")
            duration = time.time() - start_time
            return CycleResult(
                cycle_number=cycle_num,
                heartbeat_synced=heartbeat_synced,
                status_checked=status_checked,
                forum_engaged=forum_engaged,
                task_solved=task_solved,
                task_hash=task_hash,
                solana_tx=solana_tx,
                project_updated=project_updated,
                duration=duration,
                errors=errors
            )
    except Exception as e:
        log.error(f"Status check failed: {e}")
        errors.append(str(e))

    # ACT: forum engagement
    log.info("ACT: forum engagement")
    try:
        run_forum(api, forum_comment)
        forum_engaged = True
    except Exception as e:
        log.error(f"Forum engagement failed: {e}")
        errors.append(str(e))

    # THINK: task solving
    log.info("THINK: task solving")
    tasks = _load_tasks()
    task = _pick_task(tasks, cycle_num)
    result = ""
    if task:
        try:
            result = solve(task.get("description", ""))
            task_hash = hashlib.sha256(result.encode("utf-8")).hexdigest()
            task_solved = True
            log.info(f"Task solved, hash: {task_hash[:16]}...")
        except Exception as e:
            log.error(f"Task solving failed: {e}")
            errors.append(str(e))
    else:
        log.warn("No tasks found.")

    # ACT: submit proof on Solana (if configured)
    if task_hash and solana is not None:
        log.info("ACT: submit proof on Solana")
        try:
            solana_tx = solana.submit_proof(task_hash)
            log.info(f"Submitted TX: {solana_tx}")
        except Exception as e:
            log.error(f"Solana submission failed: {e}")
            errors.append(str(e))
    elif task_hash:
        log.warn("ACT: Solana not configured, skipping on-chain submission")
        log.info(f"Would submit proof hash: {task_hash[:32]}...")

    # VERIFY: transaction status (if we have a tx)
    if solana_tx and solana is not None:
        log.info("VERIFY: transaction status")
        try:
            verified = solana.verify_signature(solana_tx)
            log.info(f"Verify status: {verified}")
        except Exception as e:
            log.error(f"Verification failed: {e}")
            errors.append(str(e))

    # Update project
    try:
        ensure_project(api)
        update_text = build_update_text(task_hash or "none", solana_tx or "none")
        api.update_project(update_text)
        project_updated = True
        log.info("Project updated")
    except Exception as e:
        log.error(f"Project update failed: {e}")
        errors.append(str(e))

    duration = time.time() - start_time
    
    return CycleResult(
        cycle_number=cycle_num,
        heartbeat_synced=heartbeat_synced,
        status_checked=status_checked,
        forum_engaged=forum_engaged,
        task_solved=task_solved,
        task_hash=task_hash,
        solana_tx=solana_tx,
        project_updated=project_updated,
        duration=duration,
        errors=errors
    )


async def forever() -> None:
    """Run the agent loop forever."""
    log = get_logger("loop")
    
    log.info("="*50)
    log.info("PROOF-OF-WORK AGENT STARTING")
    log.info("="*50)
    
    api = ColosseumAPI()
    solana = get_solana_client()  # May return None if not configured
    
    if solana is None:
        log.warn("Solana client not configured - running in TEST MODE")
        log.warn("Set PROGRAM_ID to enable on-chain proof submission")
    else:
        log.info("Solana client initialized - on-chain mode enabled")
    
    cycle_num = 0
    interval = config.agent.loop_interval if hasattr(config, 'agent') else 1800
    
    log.info(f"Loop interval: {interval}s")
    log.info("Press Ctrl+C to stop")
    
    while True:
        cycle_num += 1
        log.info(f"{'='*50}")
        log.info(f"CYCLE {cycle_num} STARTED")
        log.info(f"{'='*50}")
        
        try:
            result = await run_cycle(api, solana, cycle_num)
            
            log.info(
                f"Cycle {cycle_num} complete: "
                f"heartbeat={result.heartbeat_synced}, "
                f"forum={result.forum_engaged}, "
                f"task={result.task_solved}, "
                f"tx={result.solana_tx is not None}"
            )
            
            if result.errors:
                for error in result.errors:
                    log.warn(f"Error: {error}")
                    
        except Exception as exc:
            log.error(f"Cycle error: {exc}")
        
        log.info(f"CYCLE {cycle_num} ENDED")
        log.info(f"Sleeping for {interval}s...")
        log.info(f"{'='*50}")
        
        await asyncio.sleep(interval)


class AgentLoop:
    """Object-oriented agent loop wrapper."""
    
    MODULE = "loop"
    
    def __init__(self):
        self.heartbeat_checker = HeartbeatChecker()
        self.decision = DecisionEngine()
        self.cycle_count = 0
        self.running = False
        self.log = get_logger(self.MODULE)
        
        self._api = None
        self._forum = None
        self._solana = None
    
    @property
    def api(self) -> ColosseumAPI:
        if self._api is None:
            self._api = ColosseumAPI()
        return self._api
    
    @property
    def forum(self) -> ForumHandler:
        if self._forum is None:
            self._forum = ForumHandler(self.api, self.decision)
        return self._forum
    
    @property
    def solana(self):
        """Get Solana client, or None if not configured."""
        if self._solana is None:
            self._solana = get_solana_client()
        return self._solana
    
    async def run_cycle(self) -> CycleResult:
        self.cycle_count += 1
        return await run_cycle(self.api, self.solana, self.cycle_count)
    
    async def forever(self):
        self.running = True
        await forever()
    
    def stop(self):
        self.running = False
        self.log.info("Stop signal received")
