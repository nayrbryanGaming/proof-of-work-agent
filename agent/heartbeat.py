"""
Heartbeat checker for the Proof-of-Work Agent.
Fetches and parses the heartbeat.md file from Colosseum.
"""

from __future__ import annotations

import re
import asyncio
from dataclasses import dataclass
from typing import Optional, List
import aiohttp
import requests

from agent.config import config
from agent.logger import get_logger

HEARTBEAT_URL = "https://colosseum.com/heartbeat.md"


@dataclass
class HeartbeatItem:
    """A single checklist item from heartbeat.md."""
    text: str
    completed: bool
    category: Optional[str] = None


@dataclass
class HeartbeatStatus:
    """Parsed heartbeat status."""
    raw_content: str
    items: list[HeartbeatItem]
    total_items: int
    completed_items: int
    last_updated: Optional[str] = None
    
    @property
    def completion_percentage(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    def get_pending_items(self) -> list[HeartbeatItem]:
        return [item for item in self.items if not item.completed]
    
    def get_completed_items(self) -> list[HeartbeatItem]:
        return [item for item in self.items if item.completed]


def fetch_heartbeat() -> list[str]:
    """Synchronous fetch of heartbeat items."""
    log = get_logger("heartbeat")
    url = config.agent.heartbeat_url if hasattr(config, 'agent') else HEARTBEAT_URL
    
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        items = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
        return items
    except Exception as e:
        log.warn(f"Failed to fetch heartbeat: {e}")
        return []


def check() -> list[str]:
    """Synchronous heartbeat check."""
    log = get_logger("heartbeat")
    items = fetch_heartbeat()
    for item in items:
        log.info(f"Heartbeat item: {item}")
    return items


class HeartbeatChecker:
    """Async heartbeat checker that fetches and parses heartbeat.md from Colosseum."""
    
    MODULE = "heartbeat"
    
    CHECKLIST_PATTERN = re.compile(r"^[-*]\s*\[([xX\s])\]\s*(.+)$", re.MULTILINE)
    SIMPLE_ITEM_PATTERN = re.compile(r"^[-*]\s+(.+)$", re.MULTILINE)
    CATEGORY_PATTERN = re.compile(r"^#+\s*(.+)$", re.MULTILINE)
    
    def __init__(self):
        self.heartbeat_url = config.agent.heartbeat_url if hasattr(config, 'agent') else HEARTBEAT_URL
        self._last_status: Optional[HeartbeatStatus] = None
        self.log = get_logger(self.MODULE)
    
    async def fetch_raw(self) -> str:
        """Fetch raw heartbeat.md content."""
        self.log.debug(f"Fetching heartbeat from {self.heartbeat_url}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    self.heartbeat_url,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    content = await response.text()
                    self.log.debug(f"Fetched {len(content)} bytes")
                    return content
            except aiohttp.ClientError as e:
                self.log.error(f"Failed to fetch heartbeat: {e}")
                raise
    
    def parse_content(self, content: str) -> HeartbeatStatus:
        """Parse heartbeat.md content into structured data."""
        items: list[HeartbeatItem] = []
        current_category: Optional[str] = None
        
        lines = content.split("\n")
        
        for line in lines:
            line = line.strip()
            
            category_match = self.CATEGORY_PATTERN.match(line)
            if category_match:
                current_category = category_match.group(1).strip()
                continue
            
            checklist_match = self.CHECKLIST_PATTERN.match(line)
            if checklist_match:
                completed = checklist_match.group(1).lower() == "x"
                text = checklist_match.group(2).strip()
                items.append(HeartbeatItem(
                    text=text,
                    completed=completed,
                    category=current_category
                ))
                continue
            
            simple_match = self.SIMPLE_ITEM_PATTERN.match(line)
            if simple_match:
                text = simple_match.group(1).strip()
                if not text.startswith("http") and len(text) > 3:
                    items.append(HeartbeatItem(
                        text=text,
                        completed=False,
                        category=current_category
                    ))
        
        completed_count = sum(1 for item in items if item.completed)
        
        return HeartbeatStatus(
            raw_content=content,
            items=items,
            total_items=len(items),
            completed_items=completed_count
        )
    
    async def check(self) -> HeartbeatStatus:
        """Fetch and parse heartbeat status."""
        self.log.info("Checking heartbeat...")
        
        try:
            content = await self.fetch_raw()
            status = self.parse_content(content)
            self._last_status = status
            
            self.log.success(
                f"Heartbeat synced: {status.completed_items}/{status.total_items} "
                f"items ({status.completion_percentage:.1f}% complete)"
            )
            
            pending = status.get_pending_items()
            if pending:
                self.log.debug(f"Pending items: {len(pending)}")
                for item in pending[:5]:
                    self.log.debug(f"  - {item.text[:50]}...")
            
            return status
            
        except Exception as e:
            self.log.error(f"Heartbeat check failed: {e}")
            return HeartbeatStatus(
                raw_content="",
                items=[],
                total_items=0,
                completed_items=0
            )
    
    def check_sync(self) -> list[str]:
        """Synchronous heartbeat check for backward compatibility."""
        return check()
    
    def get_last_status(self) -> Optional[HeartbeatStatus]:
        return self._last_status
    
    def get_actionable_items(self) -> list[HeartbeatItem]:
        if not self._last_status:
            return []
        
        actionable_keywords = ["submit", "deploy", "create", "update", "post", "vote"]
        pending = self._last_status.get_pending_items()
        
        return [
            item for item in pending
            if any(kw in item.text.lower() for kw in actionable_keywords)
        ]
