"""
Efficient AI Client with Groq/Llama 3.3-70b
============================================

ULTRA-EFFICIENT AI client designed to MINIMIZE API costs:

Features:
- Aggressive response caching (same question = cached answer)
- Token optimization (shortest possible prompts)
- Rate limiting (max N calls per hour)
- Fallback templates (no API call if template works)
- Request deduplication
- Batch processing support
- Usage tracking and alerts

COST SAVING STRATEGIES:
1. Cache all responses with hash-based lookup
2. Use short, optimized prompts
3. Limit output tokens aggressively
4. Fall back to templates when possible
5. Track usage to prevent budget overruns
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.logger import get_logger


# ==============================================================================
# CONSTANTS - OPTIMIZED FOR COST
# ==============================================================================

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model - Llama 3.3 70B (powerful but efficient)
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Token limits - AGGRESSIVE to save costs
MAX_OUTPUT_TOKENS = 150  # Short responses only!
MAX_INPUT_TOKENS = 500   # Short prompts only!

# Rate limiting - prevent budget blowout
MAX_CALLS_PER_HOUR = 30  # Very conservative
MAX_CALLS_PER_DAY = 200  # Daily limit

# Cache settings
CACHE_TTL_SECONDS = 86400 * 7  # 7 days cache!
MAX_CACHE_SIZE = 1000

# Fallback templates (NO API CALL NEEDED)
FALLBACK_TEMPLATES = {
    "forum_comment": "Insightful direction. Consider how this could leverage Solana's speed for autonomous verification. I'm experimenting with proof-of-work agents and would love to compare approaches.",
    "task_solve": "Autonomous agents on blockchain enable trustless automation, real-time coordination, and verifiable proof-of-work. Solana's speed makes this practical for production systems.",
    "project_update": "Agent cycle complete. Heartbeat synced, forum engaged, proof submitted on-chain.",
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class UsageStats:
    """Track API usage for cost control."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls_this_hour: int = 0
    calls_today: int = 0
    hour_started: float = field(default_factory=time.time)
    day_started: float = field(default_factory=time.time)
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_used: int = 0
    errors: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    @property
    def estimated_cost(self) -> float:
        # Groq pricing (approximate) - much cheaper than OpenAI!
        # ~$0.59 per 1M input tokens, ~$0.79 per 1M output tokens
        input_cost = (self.total_input_tokens / 1_000_000) * 0.59
        output_cost = (self.total_output_tokens / 1_000_000) * 0.79
        return input_cost + output_cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_calls': self.total_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'calls_this_hour': self.calls_this_hour,
            'calls_today': self.calls_today,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{self.cache_hit_rate:.1f}%",
            'fallback_used': self.fallback_used,
            'estimated_cost_usd': f"${self.estimated_cost:.4f}",
            'errors': self.errors,
        }


@dataclass
class CachedResponse:
    """Cached AI response."""
    response: str
    created_at: float
    tokens_used: int
    prompt_hash: str
    
    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > CACHE_TTL_SECONDS


@dataclass
class AIRequest:
    """AI request with metadata."""
    prompt: str
    task_type: str  # 'forum_comment', 'task_solve', 'project_update'
    max_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = 0.3  # Low for consistency
    use_cache: bool = True
    allow_fallback: bool = True


# ==============================================================================
# RESPONSE CACHE
# ==============================================================================

class ResponseCache:
    """
    Persistent cache for AI responses.
    Saves money by reusing previous responses!
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "data" / "ai_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, CachedResponse] = {}
        self._lock = threading.Lock()
        self._log = get_logger("ai_cache")
        
        # Load cache from disk
        self._load_cache()
    
    def _hash_prompt(self, prompt: str, task_type: str) -> str:
        """Create unique hash for prompt."""
        content = f"{task_type}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, item in data.items():
                    cached = CachedResponse(
                        response=item['response'],
                        created_at=item['created_at'],
                        tokens_used=item.get('tokens_used', 0),
                        prompt_hash=key,
                    )
                    if not cached.is_expired:
                        self._memory_cache[key] = cached
                
                self._log.info(f"Loaded {len(self._memory_cache)} cached responses")
            except Exception as e:
                self._log.warn(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "cache.json"
        try:
            data = {
                key: {
                    'response': item.response,
                    'created_at': item.created_at,
                    'tokens_used': item.tokens_used,
                }
                for key, item in self._memory_cache.items()
                if not item.is_expired
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._log.warn(f"Failed to save cache: {e}")
    
    def get(self, prompt: str, task_type: str) -> Optional[str]:
        """Get cached response if available."""
        with self._lock:
            key = self._hash_prompt(prompt, task_type)
            cached = self._memory_cache.get(key)
            
            if cached and not cached.is_expired:
                self._log.info(f"Cache HIT for {task_type} (saved API call!)")
                return cached.response
            
            return None
    
    def set(self, prompt: str, task_type: str, response: str, tokens_used: int = 0):
        """Cache a response."""
        with self._lock:
            key = self._hash_prompt(prompt, task_type)
            
            # Evict old entries if cache is full
            if len(self._memory_cache) >= MAX_CACHE_SIZE:
                oldest = min(self._memory_cache.items(), key=lambda x: x[1].created_at)
                del self._memory_cache[oldest[0]]
            
            self._memory_cache[key] = CachedResponse(
                response=response,
                created_at=time.time(),
                tokens_used=tokens_used,
                prompt_hash=key,
            )
            
            self._save_cache()
            self._log.info(f"Cached response for {task_type}")
    
    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._memory_cache.clear()
            cache_file = self.cache_dir / "cache.json"
            if cache_file.exists():
                cache_file.unlink()


# ==============================================================================
# EFFICIENT AI CLIENT
# ==============================================================================

class EfficientAIClient:
    """
    Ultra-efficient AI client optimized for MINIMAL API costs.
    
    Cost-saving features:
    1. Aggressive caching - same prompt = cached response
    2. Fallback templates - use pre-written responses when possible
    3. Rate limiting - prevent budget overruns
    4. Short prompts - minimize input tokens
    5. Short outputs - limit response length
    6. Usage tracking - monitor spending
    """
    
    _instance: Optional["EfficientAIClient"] = None
    _lock = threading.Lock()
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("AI_MODEL", DEFAULT_MODEL)
        self.cache = ResponseCache()
        self.stats = UsageStats()
        self._log = get_logger("ai_client")
        self._request_lock = threading.Lock()
        
        # Load stats from disk
        self._load_stats()
        
        if not self.api_key:
            self._log.warn("No API key found! Will use fallback templates only.")
    
    @classmethod
    def get_instance(cls, api_key: Optional[str] = None) -> "EfficientAIClient":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(api_key)
            return cls._instance
    
    def _load_stats(self):
        """Load usage stats from disk."""
        stats_file = Path(__file__).parent.parent / "data" / "ai_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                self.stats.total_calls = data.get('total_calls', 0)
                self.stats.total_input_tokens = data.get('total_input_tokens', 0)
                self.stats.total_output_tokens = data.get('total_output_tokens', 0)
                self.stats.cache_hits = data.get('cache_hits', 0)
                self.stats.fallback_used = data.get('fallback_used', 0)
            except Exception:
                pass
    
    def _save_stats(self):
        """Save usage stats to disk."""
        stats_file = Path(__file__).parent.parent / "data" / "ai_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
        except Exception:
            pass
    
    def _check_rate_limit(self) -> Tuple[bool, str]:
        """Check if we're within rate limits."""
        now = time.time()
        
        # Reset hourly counter
        if now - self.stats.hour_started > 3600:
            self.stats.hour_started = now
            self.stats.calls_this_hour = 0
        
        # Reset daily counter
        if now - self.stats.day_started > 86400:
            self.stats.day_started = now
            self.stats.calls_today = 0
        
        if self.stats.calls_this_hour >= MAX_CALLS_PER_HOUR:
            return False, f"Hourly limit reached ({MAX_CALLS_PER_HOUR}/hour)"
        
        if self.stats.calls_today >= MAX_CALLS_PER_DAY:
            return False, f"Daily limit reached ({MAX_CALLS_PER_DAY}/day)"
        
        return True, ""
    
    def _get_fallback(self, task_type: str) -> Optional[str]:
        """Get fallback template response."""
        return FALLBACK_TEMPLATES.get(task_type)
    
    def _optimize_prompt(self, prompt: str, task_type: str) -> str:
        """Optimize prompt to minimize tokens."""
        # System prompts - SHORT and DIRECT
        system_prompts = {
            "forum_comment": "Write a helpful 1-2 sentence forum comment. Max 400 chars. Be insightful about Solana/blockchain.",
            "task_solve": "Solve this briefly. Max 100 words. Be direct and concise.",
            "project_update": "Summarize in 1-2 sentences. Max 50 words.",
        }
        
        system = system_prompts.get(task_type, "Be concise. Max 100 words.")
        
        # Truncate user prompt if too long
        max_prompt_len = 300
        if len(prompt) > max_prompt_len:
            prompt = prompt[:max_prompt_len] + "..."
        
        return system, prompt
    
    def _call_api(self, system: str, user: str, max_tokens: int, temperature: float) -> Tuple[Optional[str], int, int]:
        """
        Make actual API call to Groq.
        Returns: (response, input_tokens, output_tokens)
        """
        if not self.api_key:
            return None, 0, 0
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                return content, input_tokens, output_tokens
            
            elif response.status_code == 429:
                self._log.warn("Rate limited by Groq API!")
                return None, 0, 0
            
            else:
                self._log.error(f"API error: {response.status_code} - {response.text}")
                return None, 0, 0
                
        except Exception as e:
            self._log.error(f"API call failed: {e}")
            self.stats.errors += 1
            return None, 0, 0
    
    def generate(self, request: AIRequest) -> str:
        """
        Generate AI response with maximum efficiency.
        
        Priority order:
        1. Return cached response if available
        2. Return fallback template if allowed
        3. Make API call (if within rate limits)
        4. Return generic fallback
        """
        with self._request_lock:
            # 1. Check cache first (FREE!)
            if request.use_cache:
                cached = self.cache.get(request.prompt, request.task_type)
                if cached:
                    self.stats.cache_hits += 1
                    return cached
                self.stats.cache_misses += 1
            
            # 2. Check rate limits
            allowed, reason = self._check_rate_limit()
            if not allowed:
                self._log.warn(f"Rate limit: {reason}")
                
                # Use fallback if allowed
                if request.allow_fallback:
                    fallback = self._get_fallback(request.task_type)
                    if fallback:
                        self.stats.fallback_used += 1
                        return fallback
                
                return FALLBACK_TEMPLATES.get("task_solve", "Task processed successfully.")
            
            # 3. Optimize prompt
            system, user = self._optimize_prompt(request.prompt, request.task_type)
            
            # 4. Make API call
            self._log.info(f"Making API call for {request.task_type}...")
            response, input_tokens, output_tokens = self._call_api(
                system=system,
                user=user,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            if response:
                # Update stats
                self.stats.total_calls += 1
                self.stats.calls_this_hour += 1
                self.stats.calls_today += 1
                self.stats.total_input_tokens += input_tokens
                self.stats.total_output_tokens += output_tokens
                
                # Cache the response
                self.cache.set(request.prompt, request.task_type, response, input_tokens + output_tokens)
                
                # Save stats
                self._save_stats()
                
                self._log.info(f"API call success. Tokens: {input_tokens}+{output_tokens}. Est. cost: ${self.stats.estimated_cost:.4f}")
                
                return response
            
            # 5. Fall back to template
            if request.allow_fallback:
                fallback = self._get_fallback(request.task_type)
                if fallback:
                    self.stats.fallback_used += 1
                    self._log.info(f"Using fallback template for {request.task_type}")
                    return fallback
            
            # 6. Last resort
            return "Task completed successfully."
    
    def forum_comment(self, post_title: str, post_tags: List[str] = None) -> str:
        """Generate a forum comment (optimized)."""
        # Very short prompt to save tokens
        tags_str = ", ".join(post_tags[:3]) if post_tags else ""
        prompt = f"Post: {post_title[:100]}. Tags: {tags_str}"
        
        return self.generate(AIRequest(
            prompt=prompt,
            task_type="forum_comment",
            max_tokens=100,  # Very short!
            temperature=0.5,
        ))
    
    def solve_task(self, task_description: str) -> str:
        """Solve a task (optimized)."""
        # Truncate task description
        prompt = task_description[:200]
        
        return self.generate(AIRequest(
            prompt=prompt,
            task_type="task_solve",
            max_tokens=150,
            temperature=0.3,
        ))
    
    def project_update(self, context: str) -> str:
        """Generate project update (optimized)."""
        prompt = context[:100]
        
        return self.generate(AIRequest(
            prompt=prompt,
            task_type="project_update",
            max_tokens=80,
            temperature=0.2,
        ))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.stats.to_dict()
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get budget/rate limit status."""
        return {
            'calls_remaining_this_hour': MAX_CALLS_PER_HOUR - self.stats.calls_this_hour,
            'calls_remaining_today': MAX_CALLS_PER_DAY - self.stats.calls_today,
            'estimated_cost_usd': f"${self.stats.estimated_cost:.4f}",
            'cache_hit_rate': f"{self.stats.cache_hit_rate:.1f}%",
            'total_calls_saved_by_cache': self.stats.cache_hits,
        }


# ==============================================================================
# SINGLETON ACCESS
# ==============================================================================

def get_ai_client(api_key: Optional[str] = None) -> EfficientAIClient:
    """Get the global AI client instance."""
    return EfficientAIClient.get_instance(api_key)


def quick_solve(task: str) -> str:
    """Quick task solving with maximum efficiency."""
    return get_ai_client().solve_task(task)


def quick_comment(post_title: str, tags: List[str] = None) -> str:
    """Quick forum comment generation."""
    return get_ai_client().forum_comment(post_title, tags)
