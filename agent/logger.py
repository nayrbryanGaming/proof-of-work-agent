"""
Structured logging for the Proof-of-Work Agent.
Format: [TIME][LEVEL][MODULE] message
Levels: INFO/WARN/ERROR
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import threading

from rich.console import Console
from rich.logging import RichHandler


_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "agent.log"
_LOCK = threading.Lock()


def _ensure_log_path() -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_PATH.exists():
        _LOG_PATH.touch()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.upper()[:4]
        module = record.name.split(".")[-1] if "." in record.name else record.name
        message = record.getMessage()
        
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{exc_text}"
        
        return f"[{timestamp}][{level}][{module}] {message}"


class Logger:
    """Simple structured logger for the agent."""
    
    def __init__(self, module: str) -> None:
        self.module = module
        _ensure_log_path()
        self._logger = logging.getLogger(f"pow_agent.{module}")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup file and console handlers."""
        if self._logger.handlers:
            return
            
        self._logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        console_handler.setLevel(logging.INFO)
        self._logger.addHandler(console_handler)

    def _log(self, level: str, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}][{level}][{self.module}] {message}"
        with _LOCK:
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(line)

    def info(self, message: str) -> None:
        self._log("INFO", message)

    def warn(self, message: str) -> None:
        self._log("WARN", message)

    def error(self, message: str, exc_info: bool = False) -> None:
        self._log("ERROR", message)
        if exc_info:
            import traceback
            tb = traceback.format_exc()
            if tb and tb != "NoneType: None\n":
                with _LOCK:
                    with _LOG_PATH.open("a", encoding="utf-8") as f:
                        f.write(tb + "\n")

    def debug(self, message: str) -> None:
        self._log("DEBUG", message)

    def success(self, message: str) -> None:
        self._log("INFO", f"✓ {message}")

    def action(self, action: str, details: str = "") -> None:
        msg = f"ACTION: {action}"
        if details:
            msg += f" | {details}"
        self._log("INFO", msg)


def get_logger(module: str) -> Logger:
    """Get a logger for a specific module."""
    return Logger(module)


class AgentLogger:
    """Enhanced structured logger for the agent with Rich support."""
    
    _instance: Optional["AgentLogger"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "AgentLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if AgentLogger._initialized:
            return
        
        AgentLogger._initialized = True
        self.console = Console()
        self._loggers: dict[str, Logger] = {}
        _ensure_log_path()
    
    def get_logger(self, module: str) -> Logger:
        """Get a logger for a specific module."""
        if module not in self._loggers:
            self._loggers[module] = Logger(module)
        return self._loggers[module]
    
    def info(self, module: str, message: str):
        self.get_logger(module).info(message)
    
    def warn(self, module: str, message: str):
        self.get_logger(module).warn(message)
    
    def error(self, module: str, message: str, exc_info: bool = False):
        self.get_logger(module).error(message, exc_info=exc_info)
    
    def debug(self, module: str, message: str):
        self.get_logger(module).debug(message)
    
    def success(self, module: str, message: str):
        self.get_logger(module).success(message)
    
    def action(self, module: str, action: str, details: str = ""):
        self.get_logger(module).action(action, details)
    
    def cycle_start(self, cycle_num: int):
        self.info("loop", f"{'='*50}")
        self.info("loop", f"CYCLE {cycle_num} STARTED")
        self.info("loop", f"{'='*50}")
    
    def cycle_end(self, cycle_num: int, duration: float):
        self.info("loop", f"CYCLE {cycle_num} COMPLETED in {duration:.2f}s")
        self.info("loop", f"{'='*50}")


# Global logger instance
logger = AgentLogger()
