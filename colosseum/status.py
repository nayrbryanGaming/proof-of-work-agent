"""
Status handling for the Proof-of-Work Agent.
Processes and interprets agent status from Colosseum.
"""

from __future__ import annotations

from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum

from agent.logger import get_logger


class AgentState(Enum):
    """Possible agent states."""
    ACTIVE = "active"
    RUNNING = "running"
    IDLE = "idle"
    PAUSED = "paused"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class StatusMetrics:
    """Agent status metrics."""
    total_tasks_completed: int = 0
    total_votes_cast: int = 0
    total_comments_made: int = 0
    total_transactions: int = 0
    uptime_hours: float = 0.0
    last_active: Optional[str] = None


@dataclass
class AgentStatus:
    """Parsed agent status."""
    state: AgentState
    is_active: bool
    metrics: StatusMetrics
    raw_data: dict
    warnings: list[str]
    recommendations: list[str]


def should_act(status: dict) -> bool:
    """Determine if agent should act based on status.
    
    Returns True if agent should perform actions this cycle.
    Always returns True unless explicitly told to pause/stop.
    """
    if not isinstance(status, dict):
        return True
    
    # Check for explicit pause/stop signals
    state = status.get("status")
    if isinstance(status.get("agent"), dict) and not state:
        state = status["agent"].get("status")
    
    # Only return False if explicitly paused/stopped
    if state:
        state_lower = str(state).lower()
        if state_lower in ("paused", "stopped", "disabled", "banned"):
            return False
    
    # Check nextSteps for pause signals
    next_steps = status.get("nextSteps")
    if isinstance(next_steps, list):
        joined = " ".join([str(s).lower() for s in next_steps])
        if any(k in joined for k in ["pause", "wait", "stop"]):
            return False
    
    # Default: always act!
    return True


class StatusChecker:
    """Processes and interprets agent status."""
    
    MODULE = "status"
    
    def __init__(self, api):
        self.api = api
        self.log = get_logger(self.MODULE)
        self._last_status: Optional[AgentStatus] = None
    
    def _parse_state(self, raw_state: Any) -> AgentState:
        if not raw_state:
            return AgentState.UNKNOWN
        
        state_str = str(raw_state).lower()
        
        for state in AgentState:
            if state.value == state_str:
                return state
        
        return AgentState.UNKNOWN
    
    def _parse_metrics(self, data: dict) -> StatusMetrics:
        metrics_data = data.get("metrics", {})
        
        return StatusMetrics(
            total_tasks_completed=metrics_data.get("tasks_completed", 0),
            total_votes_cast=metrics_data.get("votes_cast", 0),
            total_comments_made=metrics_data.get("comments_made", 0),
            total_transactions=metrics_data.get("transactions", 0),
            uptime_hours=metrics_data.get("uptime_hours", 0.0),
            last_active=metrics_data.get("last_active")
        )
    
    def _extract_warnings(self, data: dict) -> list[str]:
        warnings = []
        
        if "warnings" in data:
            warnings.extend(data["warnings"])
        
        if data.get("rate_limited"):
            warnings.append("Agent is rate limited")
        
        metrics = data.get("metrics", {})
        if metrics.get("tasks_completed", 0) == 0:
            warnings.append("No tasks completed yet")
        
        return warnings
    
    def _generate_recommendations(self, data: dict, state: AgentState) -> list[str]:
        recommendations = []
        
        if state == AgentState.PAUSED:
            recommendations.append("Resume agent activity")
        
        if state == AgentState.ERROR:
            recommendations.append("Check logs for errors")
            recommendations.append("Verify API credentials")
        
        metrics = data.get("metrics", {})
        if metrics.get("votes_cast", 0) < 5:
            recommendations.append("Increase forum engagement")
        
        return recommendations
    
    def parse_status(self, raw_data: dict) -> AgentStatus:
        """Parse raw status data into structured AgentStatus."""
        state = self._parse_state(raw_data.get("state", raw_data.get("status")))
        metrics = self._parse_metrics(raw_data)
        warnings = self._extract_warnings(raw_data)
        recommendations = self._generate_recommendations(raw_data, state)
        
        return AgentStatus(
            state=state,
            is_active=state in (AgentState.ACTIVE, AgentState.RUNNING),
            metrics=metrics,
            raw_data=raw_data,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def check(self) -> AgentStatus:
        """Fetch and parse agent status."""
        self.log.info("Checking agent status...")
        
        try:
            raw_data = self.api.get_agent_status()
            status = self.parse_status(raw_data)
            self._last_status = status
            
            self.log.success(
                f"Status: {status.state.value} | "
                f"Tasks: {status.metrics.total_tasks_completed} | "
                f"Votes: {status.metrics.total_votes_cast}"
            )
            
            for warning in status.warnings:
                self.log.warn(f"Warning: {warning}")
            
            return status
            
        except Exception as e:
            self.log.error(f"Status check failed: {e}")
            
            return AgentStatus(
                state=AgentState.UNKNOWN,
                is_active=False,
                metrics=StatusMetrics(),
                raw_data={},
                warnings=["Failed to fetch status"],
                recommendations=["Retry status check"]
            )
    
    def get_last_status(self) -> Optional[AgentStatus]:
        return self._last_status
    
    def should_proceed(self) -> bool:
        """Determine if agent should proceed with actions based on status."""
        if not self._last_status:
            return True
        
        if self._last_status.state in (AgentState.ERROR, AgentState.PAUSED):
            return False
        
        return True
