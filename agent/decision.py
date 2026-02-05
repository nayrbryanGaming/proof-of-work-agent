"""
Decision engine for the Proof-of-Work Agent.
Uses Groq/Llama 3.3-70b for COST-EFFICIENT AI responses.

OPTIMIZED FOR MINIMAL API COSTS:
- Aggressive caching
- Short prompts
- Fallback templates
- Rate limiting
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.config import config
from agent.logger import get_logger

# Import efficient AI client
try:
    from agent.ai_client import get_ai_client, EfficientAIClient, AIRequest
    HAS_AI_CLIENT = True
except ImportError:
    HAS_AI_CLIENT = False

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "solve_task.txt"
_DEFAULT_MODELS = ["llama-3.3-70b-versatile", "gpt-4", "gpt-3.5-turbo"]

# Fallback responses (NO API CALL - FREE!)
FALLBACK_RESPONSES = {
    "forum_comment": "Insightful direction. Consider how this could leverage Solana's speed for autonomous verification. I'm experimenting with proof-of-work agents and would love to compare approaches.",
    "task_solve": "Autonomous agents in blockchain enable trustless automation through cryptographic verification. Solana's sub-second finality makes real-time agent coordination possible, allowing for proof-of-work validation without centralized intermediaries.",
}


class OpenAIRetryableError(Exception):
    pass


@dataclass
class TaskResult:
    """Result of solving a task."""
    task_id: int
    task_description: str
    solution: str
    proof_hash: str
    success: bool
    error: Optional[str] = None


@dataclass
class ForumContext:
    """Context for generating forum comments."""
    post_id: str
    post_title: str
    post_content: str
    post_tags: list[str]
    existing_comments: list[str]


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return "You are a helpful assistant."


def _compute_proof_hash(result: str) -> str:
    """Compute SHA256 hash of a result as proof of work."""
    return hashlib.sha256(result.encode("utf-8")).hexdigest()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(OpenAIRetryableError),
)
def _post_openai(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    if resp.status_code in (429, 500, 502, 503, 504):
        raise OpenAIRetryableError(f"OpenAI retryable error: {resp.status_code}")
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}")
    return resp.json()


def _call_openai(messages: list[dict], max_tokens: int = 400) -> str:
    log = get_logger("decision")
    if not config.openai_api_key:
        log.warn("OPENAI_API_KEY missing, using fallback response.")
        return "OpenAI key missing. Provide a key to enable full reasoning."
    
    env_model = os.getenv("OPENAI_MODEL", "").strip()
    model_list = [env_model] if env_model else []
    model_list.extend([m for m in _DEFAULT_MODELS if m not in model_list])
    
    last_error = None
    for model in model_list:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            response = _post_openai(payload)
            text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text:
                return text.strip()
        except Exception as exc:
            last_error = exc
            log.warn(f"OpenAI model {model} failed: {exc}")
            continue
    
    raise RuntimeError(f"OpenAI request failed: {last_error}")


def solve(task: str) -> str:
    """
    Solve a task using AI (COST-OPTIMIZED).
    
    Uses efficient AI client with caching to minimize API calls.
    """
    log = get_logger("decision")
    
    # Try efficient AI client first (with caching!)
    if HAS_AI_CLIENT:
        try:
            client = get_ai_client()
            result = client.solve_task(task)
            log.info(f"Task solved via efficient AI client. Stats: {client.get_budget_status()}")
            return result
        except Exception as e:
            log.warn(f"Efficient AI client failed: {e}, trying fallback...")
    
    # Fallback to OpenAI if configured
    if config.openai_api_key:
        system_prompt = _load_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task:\n{task}\n\nRespond with the solution only."},
        ]
        try:
            return _call_openai(messages, max_tokens=150)  # Reduced tokens!
        except Exception as e:
            log.warn(f"OpenAI fallback failed: {e}")
    
    # Final fallback - FREE!
    log.info("Using fallback template (no API cost)")
    return FALLBACK_RESPONSES["task_solve"]


def _fallback_comment(context: str) -> str:
    title = ""
    for line in context.splitlines():
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            break
    if title:
        text = f'Thanks for sharing "{title}". Happy to review or test when you post more details.'
    else:
        text = "Thanks for sharing this update. Happy to review or test when you post more details."
    return text[:400]


def forum_comment(context: str) -> str:
    """
    Generate a forum comment (COST-OPTIMIZED).
    
    Uses caching and fallback to minimize API calls.
    """
    log = get_logger("decision")
    
    # Extract title and tags for efficient AI client
    title = ""
    tags = []
    for line in context.splitlines():
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("tags:"):
            tags = [t.strip() for t in line.split(":", 1)[1].split(",")]
    
    # Try efficient AI client first (with caching!)
    if HAS_AI_CLIENT:
        try:
            client = get_ai_client()
            result = client.forum_comment(title or context[:100], tags)
            
            # Ensure max 400 chars
            if len(result) > 400:
                result = result[:397].rstrip() + "..."
            
            log.info(f"Comment generated via efficient AI. Budget: {client.get_budget_status()}")
            return result
        except Exception as e:
            log.warn(f"Efficient AI client failed: {e}")
    
    # Fallback to OpenAI if configured
    if config.openai_api_key:
        system_prompt = (
            "Write one concise, helpful comment for a Solana hackathon forum post. "
            "Be specific to the context. 1-2 sentences. Max 400 characters."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context[:300]},  # Truncated!
        ]
        try:
            text = _call_openai(messages, max_tokens=100)  # Reduced tokens!
            text = " ".join(text.strip().splitlines()).strip()
            if len(text) > 400:
                text = text[:397].rstrip() + "..."
            return text
        except Exception as e:
            log.warn(f"OpenAI fallback failed: {e}")
    
    # Final fallback - FREE!
    log.info("Using fallback comment template (no API cost)")
    return FALLBACK_RESPONSES["forum_comment"]


class DecisionEngine:
    """AI-powered decision engine for the agent."""
    
    MODULE = "decision"
    MAX_COMMENT_LENGTH = 400
    
    def __init__(self):
        self.log = get_logger(self.MODULE)
        self.prompts_dir = config.get_prompts_dir() if hasattr(config, 'get_prompts_dir') else _PROMPT_PATH.parent
        self._prompt_cache: dict[str, str] = {}
    
    def _load_prompt(self, name: str) -> str:
        if name in self._prompt_cache:
            return self._prompt_cache[name]
        
        prompt_file = self.prompts_dir / f"{name}.txt"
        if not prompt_file.exists():
            self.log.warn(f"Prompt file not found: {prompt_file}")
            return ""
        
        content = prompt_file.read_text(encoding="utf-8")
        self._prompt_cache[name] = content
        return content
    
    def solve(self, task: dict) -> TaskResult:
        """Solve a task using AI and return the result with proof hash."""
        task_id = task.get("id", 0)
        description = task.get("description", "")
        
        self.log.info(f"Solving task {task_id}: {description[:50]}...")
        
        try:
            solution = solve(description)
            proof_hash = _compute_proof_hash(solution)
            
            self.log.success(f"Task {task_id} solved, hash: {proof_hash[:16]}...")
            
            return TaskResult(
                task_id=task_id,
                task_description=description,
                solution=solution,
                proof_hash=proof_hash,
                success=True
            )
            
        except Exception as e:
            self.log.error(f"Failed to solve task {task_id}: {e}")
            return TaskResult(
                task_id=task_id,
                task_description=description,
                solution="",
                proof_hash="",
                success=False,
                error=str(e)
            )
    
    def forum_comment(self, context: ForumContext) -> str:
        """Generate a helpful comment for a forum post."""
        self.log.debug(f"Generating comment for post: {context.post_title[:30]}...")
        
        tags_str = ", ".join(context.post_tags) if context.post_tags else "general"
        context_str = f"Title: {context.post_title}\nTags: {tags_str}\nBody: {context.post_content[:500]}"
        
        return forum_comment(context_str)
    
    def analyze_status(self, status_data: dict) -> dict:
        """Analyze agent status and determine actions."""
        self.log.debug("Analyzing agent status...")
        
        return {
            "should_engage_forum": True,
            "should_solve_tasks": True,
            "should_update_project": True,
            "priority_actions": ["continue_normal_operation"],
            "reasoning": "Proceeding with default actions"
        }
    
    def generate_project_update(
        self,
        heartbeat_status: str,
        forum_activity: str,
        task_hash: str,
        solana_tx: str
    ) -> str:
        """Generate a project update message."""
        return f"""Autonomous report:
- Heartbeat synced: {heartbeat_status}
- Forum engaged: {forum_activity}
- Task solved hash: {task_hash}
- Solana TX: {solana_tx}

Generated by POW-Agent at cycle completion."""
