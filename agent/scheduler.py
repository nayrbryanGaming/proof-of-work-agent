"""
Scheduler - Cron-like scheduling system.
Supports complex schedules, intervals, and one-time tasks.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from threading import Lock
import calendar

from agent.logger import get_logger


class ScheduleType(Enum):
    """Types of schedules."""
    INTERVAL = "interval"
    CRON = "cron"
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ScheduledJob:
    """A scheduled job definition."""
    id: str
    name: str
    func: Callable
    schedule_type: ScheduleType
    schedule_data: Dict[str, Any]
    
    # Execution tracking
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    
    # Configuration
    enabled: bool = True
    max_instances: int = 1
    coalesce: bool = True  # Skip missed runs
    jitter: Optional[float] = None  # Random delay
    
    # State
    running_instances: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "schedule_data": self.schedule_data,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "enabled": self.enabled,
            "running": self.running_instances > 0
        }


class CronExpression:
    """
    Parses and evaluates cron expressions.
    Format: minute hour day_of_month month day_of_week
    
    Supports:
    - * (any value)
    - */n (every n)
    - n-m (range)
    - n,m,o (list)
    """
    
    FIELD_RANGES = [
        (0, 59),   # minute
        (0, 23),   # hour
        (1, 31),   # day of month
        (1, 12),   # month
        (0, 6),    # day of week (0 = Sunday)
    ]
    
    def __init__(self, expression: str):
        self.expression = expression
        self.fields = self._parse(expression)
    
    def _parse(self, expression: str) -> List[Set[int]]:
        """Parse cron expression into sets of valid values."""
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")
        
        fields = []
        for i, part in enumerate(parts):
            min_val, max_val = self.FIELD_RANGES[i]
            fields.append(self._parse_field(part, min_val, max_val))
        
        return fields
    
    def _parse_field(self, field: str, min_val: int, max_val: int) -> Set[int]:
        """Parse a single cron field."""
        values = set()
        
        for part in field.split(','):
            if part == '*':
                values.update(range(min_val, max_val + 1))
            elif '/' in part:
                base, step = part.split('/')
                step = int(step)
                if base == '*':
                    start = min_val
                else:
                    start = int(base)
                values.update(range(start, max_val + 1, step))
            elif '-' in part:
                start, end = map(int, part.split('-'))
                values.update(range(start, end + 1))
            else:
                values.add(int(part))
        
        return values
    
    def matches(self, dt: datetime) -> bool:
        """Check if datetime matches cron expression."""
        minute, hour, day, month, weekday = self.fields
        
        return (
            dt.minute in minute and
            dt.hour in hour and
            dt.day in day and
            dt.month in month and
            dt.weekday() in weekday  # Python weekday (0 = Monday)
        )
    
    def get_next(self, from_dt: Optional[datetime] = None) -> datetime:
        """Get next matching datetime."""
        dt = from_dt or datetime.now(timezone.utc)
        dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Search up to 2 years ahead
        max_iterations = 2 * 365 * 24 * 60  # 2 years in minutes
        
        for _ in range(max_iterations):
            if self.matches(dt):
                return dt
            dt += timedelta(minutes=1)
        
        raise ValueError(f"No matching time found for {self.expression}")


class IntervalSchedule:
    """Simple interval-based schedule."""
    
    def __init__(self, seconds: float):
        self.seconds = seconds
    
    def get_next(self, last_run: Optional[datetime] = None) -> datetime:
        """Get next run time."""
        base = last_run or datetime.now(timezone.utc)
        return base + timedelta(seconds=self.seconds)


class Scheduler:
    """
    Job scheduler with cron-like capabilities.
    """
    
    def __init__(self):
        self.log = get_logger("scheduler")
        self._jobs: Dict[str, ScheduledJob] = {}
        self._lock = Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._job_id_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique job ID."""
        self._job_id_counter += 1
        return f"job_{self._job_id_counter}_{int(time.time())}"
    
    def add_job(
        self,
        func: Callable,
        trigger: str,
        name: Optional[str] = None,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        **trigger_args
    ) -> str:
        """
        Add a scheduled job.
        
        Args:
            func: Function to execute
            trigger: Trigger type ('interval', 'cron', 'once', 'daily')
            name: Job name
            args, kwargs: Arguments to pass to function
            **trigger_args: Trigger-specific arguments
        
        Returns:
            Job ID
        """
        job_id = self._generate_id()
        
        # Wrap function with arguments
        if args or kwargs:
            original_func = func
            async def wrapped():
                if asyncio.iscoroutinefunction(original_func):
                    return await original_func(*(args or ()), **(kwargs or {}))
                return original_func(*(args or ()), **(kwargs or {}))
            func = wrapped
        
        # Determine schedule type and calculate next run
        if trigger == 'interval':
            schedule_type = ScheduleType.INTERVAL
            seconds = trigger_args.get('seconds', 0)
            seconds += trigger_args.get('minutes', 0) * 60
            seconds += trigger_args.get('hours', 0) * 3600
            seconds += trigger_args.get('days', 0) * 86400
            schedule_data = {'seconds': seconds}
            next_run = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            
        elif trigger == 'cron':
            schedule_type = ScheduleType.CRON
            expression = trigger_args.get('expression', '* * * * *')
            schedule_data = {'expression': expression}
            cron = CronExpression(expression)
            next_run = cron.get_next()
            
        elif trigger == 'once':
            schedule_type = ScheduleType.ONCE
            run_at = trigger_args.get('run_at')
            if isinstance(run_at, str):
                run_at = datetime.fromisoformat(run_at)
            schedule_data = {'run_at': run_at.isoformat()}
            next_run = run_at
            
        elif trigger == 'daily':
            schedule_type = ScheduleType.DAILY
            hour = trigger_args.get('hour', 0)
            minute = trigger_args.get('minute', 0)
            schedule_data = {'hour': hour, 'minute': minute}
            now = datetime.now(timezone.utc)
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        
        else:
            raise ValueError(f"Unknown trigger type: {trigger}")
        
        job = ScheduledJob(
            id=job_id,
            name=name or func.__name__,
            func=func,
            schedule_type=schedule_type,
            schedule_data=schedule_data,
            next_run=next_run,
            max_instances=trigger_args.get('max_instances', 1),
            coalesce=trigger_args.get('coalesce', True),
            jitter=trigger_args.get('jitter')
        )
        
        with self._lock:
            self._jobs[job_id] = job
        
        self.log.info(f"Added job '{job.name}' ({trigger}), next run: {next_run}")
        return job_id
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                self.log.info(f"Removed job: {job_id}")
                return True
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = False
                return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = True
                return True
        return False
    
    def _calculate_next_run(self, job: ScheduledJob) -> Optional[datetime]:
        """Calculate next run time for a job."""
        if job.schedule_type == ScheduleType.INTERVAL:
            return IntervalSchedule(job.schedule_data['seconds']).get_next(job.last_run)
        
        elif job.schedule_type == ScheduleType.CRON:
            cron = CronExpression(job.schedule_data['expression'])
            return cron.get_next(job.last_run)
        
        elif job.schedule_type == ScheduleType.ONCE:
            return None  # One-time job, no next run
        
        elif job.schedule_type == ScheduleType.DAILY:
            last = job.last_run or datetime.now(timezone.utc)
            next_run = last.replace(
                hour=job.schedule_data['hour'],
                minute=job.schedule_data['minute'],
                second=0,
                microsecond=0
            )
            if next_run <= last:
                next_run += timedelta(days=1)
            return next_run
        
        return None
    
    async def _run_job(self, job: ScheduledJob):
        """Execute a job."""
        if not job.enabled:
            return
        
        if job.running_instances >= job.max_instances:
            self.log.warn(f"Job '{job.name}' max instances reached, skipping")
            return
        
        job.running_instances += 1
        
        try:
            self.log.debug(f"Running job: {job.name}")
            
            if asyncio.iscoroutinefunction(job.func):
                await job.func()
            else:
                await asyncio.to_thread(job.func)
            
            job.run_count += 1
            self.log.debug(f"Job '{job.name}' completed (run #{job.run_count})")
            
        except Exception as e:
            job.error_count += 1
            self.log.error(f"Job '{job.name}' failed: {e}")
        
        finally:
            job.running_instances -= 1
            job.last_run = datetime.now(timezone.utc)
            
            # Calculate next run
            job.next_run = self._calculate_next_run(job)
            
            # Add jitter if configured
            if job.jitter and job.next_run:
                import random
                jitter = random.uniform(-job.jitter, job.jitter)
                job.next_run += timedelta(seconds=jitter)
            
            # Remove one-time jobs
            if job.schedule_type == ScheduleType.ONCE:
                self.remove_job(job.id)
    
    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            now = datetime.now(timezone.utc)
            jobs_to_run = []
            
            with self._lock:
                for job in self._jobs.values():
                    if not job.enabled or not job.next_run:
                        continue
                    
                    if job.next_run <= now:
                        if job.coalesce:
                            # Skip missed runs, just run once
                            jobs_to_run.append(job)
                        else:
                            # Run for each missed interval (not recommended)
                            jobs_to_run.append(job)
            
            # Run due jobs concurrently
            if jobs_to_run:
                tasks = [self._run_job(job) for job in jobs_to_run]
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Sleep until next check (every second)
            await asyncio.sleep(1)
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self.log.info("Scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.log.info("Scheduler stopped")
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs."""
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None


# Global scheduler
scheduler = Scheduler()


# Convenience decorators
def interval(
    seconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
    **kwargs
):
    """Decorator to schedule function at intervals."""
    def decorator(func: Callable):
        scheduler.add_job(
            func,
            'interval',
            seconds=seconds,
            minutes=minutes,
            hours=hours,
            **kwargs
        )
        return func
    return decorator


def cron(expression: str, **kwargs):
    """Decorator to schedule function with cron expression."""
    def decorator(func: Callable):
        scheduler.add_job(func, 'cron', expression=expression, **kwargs)
        return func
    return decorator


def daily(hour: int = 0, minute: int = 0, **kwargs):
    """Decorator to schedule function daily."""
    def decorator(func: Callable):
        scheduler.add_job(func, 'daily', hour=hour, minute=minute, **kwargs)
        return func
    return decorator
