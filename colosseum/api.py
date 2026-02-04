"""
Colosseum API wrapper with retry policy.
Retry policy: 3 retries, exponential backoff on 429/5xx.
"""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.config import config
from agent.logger import get_logger

BASE_URL = "https://agents.colosseum.com/api"


class ApiError(Exception):
    """Base API error."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class RetryableError(Exception):
    """Error that should trigger a retry."""
    pass


class RateLimitError(ApiError):
    """Rate limit (429) error."""
    pass


class ServerError(ApiError):
    """Server (5xx) error."""
    pass


@dataclass
class APIResponse:
    """Structured API response."""
    success: bool
    data: Any
    status_code: int
    error: Optional[str] = None


class ColosseumAPI:
    """Colosseum API client with retry logic."""
    
    MODULE = "api"
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.colosseum_api_key
        self.base_url = config.colosseum.base_url if hasattr(config, 'colosseum') else BASE_URL
        self.session = requests.Session()
        self.log = get_logger("colosseum.api")

    def _auth_headers(self) -> dict:
        """Get request headers with authentication."""
        if not self.api_key:
            raise ValueError("COLOSSEUM_API_KEY is required.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "POW-Agent/1.0"
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RetryableError),
    )
    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an API request with retry logic."""
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update(self._auth_headers())
        
        self.log.debug(f"{method} {path}")
        
        resp = self.session.request(method, url, headers=headers, timeout=20, **kwargs)
        
        if resp.status_code == 429:
            self.log.warn("Rate limited, will retry...")
            raise RetryableError(f"Rate limited: {resp.status_code}")
        
        if resp.status_code in (500, 502, 503, 504):
            self.log.warn(f"Server error {resp.status_code}, will retry...")
            raise RetryableError(f"Server error: {resp.status_code}")
        
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, resp.text)
        
        if resp.text:
            return resp.json()
        return {}

    def get_agent_status(self) -> dict:
        """
        GET /api/agents/status
        Get current agent status from Colosseum.
        """
        self.log.info("Fetching agent status...")
        try:
            result = self._request("GET", "/agents/status")
            self.log.success("Agent status retrieved")
            return result
        except Exception as e:
            self.log.warn(f"Failed to get status: {e}")
            return {}

    def get_hot_posts(self, limit: int = 20) -> list:
        """
        GET /api/forum/posts?sort=hot&limit=20
        Get hot forum posts.
        """
        self.log.info(f"Fetching hot posts (limit={limit})...")
        try:
            data = self._request("GET", f"/forum/posts?sort=hot&limit={limit}")
            if isinstance(data, list):
                posts = data
            else:
                posts = data.get("posts", [])
            self.log.success(f"Retrieved {len(posts)} hot posts")
            return posts
        except Exception as e:
            self.log.warn(f"Failed to get posts: {e}")
            return []

    def vote_post(self, post_id: int) -> dict:
        """
        POST /api/forum/posts/{id}/vote {value:1}
        Vote on a forum post.
        """
        self.log.action("vote_post", f"post_id={post_id}")
        try:
            result = self._request("POST", f"/forum/posts/{post_id}/vote", json={"value": 1})
            self.log.success(f"Voted on post {post_id}")
            return result
        except Exception as e:
            self.log.warn(f"Failed to vote on post {post_id}: {e}")
            return {}

    def comment_post(self, post_id: int, text: str) -> dict:
        """
        POST /api/forum/posts/{id}/comments
        Comment on a forum post.
        """
        self.log.action("comment_post", f"post_id={post_id}, text_len={len(text)}")
        try:
            result = self._request(
                "POST", f"/forum/posts/{post_id}/comments", json={"body": text, "text": text}
            )
            self.log.success(f"Commented on post {post_id}")
            return result
        except Exception as e:
            self.log.warn(f"Failed to comment on post {post_id}: {e}")
            return {}

    def update_project(self, text: str) -> dict:
        """
        PUT /api/my-project
        Update the agent's project draft.
        """
        self.log.action("update_project", f"text_len={len(text)}")
        try:
            result = self._request("PUT", "/my-project", json={"description": text, "content": text})
            self.log.success("Project updated")
            return result
        except Exception as e:
            self.log.warn(f"Failed to update project: {e}")
            return {}

    def get_my_project(self) -> dict:
        """
        GET /api/my-project
        Get the agent's current project.
        """
        self.log.info("Fetching project...")
        result = self._request("GET", "/my-project")
        self.log.success("Project retrieved")
        return result

    def create_project(self, payload: dict) -> dict:
        """
        POST /api/my-project
        Create a new project.
        """
        self.log.action("create_project", f"payload_keys={list(payload.keys())}")
        result = self._request("POST", "/my-project", json=payload)
        self.log.success("Project created")
        return result
