"""
WebSocket Connection Manager for real-time updates.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from fastapi import WebSocket

from agent.logger import get_logger


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.log = get_logger("websocket")
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        self.log.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        self.log.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.log.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected = set()
        
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    self.log.error(f"Broadcast failed for connection: {e}")
                    disconnected.add(connection)
            
            # Remove disconnected clients
            self.active_connections -= disconnected
    
    async def broadcast_log(self, level: str, module: str, message: str):
        """Broadcast a log entry."""
        await self.broadcast({
            "type": "log",
            "data": {
                "level": level,
                "module": module,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    
    async def broadcast_metric(self, metric_name: str, value: Any):
        """Broadcast a metric update."""
        await self.broadcast({
            "type": "metric",
            "data": {
                "name": metric_name,
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    
    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)


# Global manager instance
manager = ConnectionManager()
