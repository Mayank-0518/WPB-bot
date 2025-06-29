"""
Scheduler for WhatsApp AI Second Brain Assistant
Handles scheduling and execution of reminders and periodic tasks

Example Usage:
# Schedule a reminder
await scheduler.schedule_reminder(
    reminder_id="rem_001",
    user_id="user123", 
    message="Don't forget to review the quarterly report",
    scheduled_time=datetime(2025, 6, 30, 14, 0)
)

# Schedule recurring task
await scheduler.schedule_recurring_task(
    task_id="daily_summary",
    user_id="user123",
    interval_hours=24,
    task_type="daily_summary"
)
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Callable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from backend.utils.config import config
from backend.scheduler.jobs import ReminderJob

logger = logging.getLogger(__name__)

class SecondBrainScheduler:
    """Async scheduler for reminders and periodic tasks"""
    
    def __init__(self):
        self.scheduler = None
        self.jobs = {}  # job_id -> job_info
        self.timezone = timezone.utc  # Force UTC timezone
        self.reminder_job = ReminderJob()
        self._started = False
    
    async def initialize(self):
        """Initialize the scheduler"""
        if self._started:
            return
        
        try:
            # Configure job stores
            jobstores = {
                'default': MemoryJobStore()
            }
            
            # Configure executors
            job_defaults = {
                'coalesce': False,
                'max_instances': 3,
                'misfire_grace_time': 300  # 5 minutes
            }
            
            # Create scheduler
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                job_defaults=job_defaults,
                timezone=self.timezone
            )
            
            # Add event listeners
            self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
            self.scheduler.add_listener(self._job_missed, EVENT_JOB_MISSED)
            
            # Start scheduler
            self.scheduler.start()
            self._started = True
            
            logger.info("Scheduler initialized and started")
            
        except Exception as e:
            logger.error(f"Scheduler initialization failed: {e}")
            raise
    
    async def schedule_reminder(
        self, 
        reminder_id: str,
        user_id: str,
        message: str,
        scheduled_time: datetime,
        phone_number: str
    ) -> str:
        """
        Schedule a one-time reminder
        
        Args:
            reminder_id: Unique reminder identifier
            user_id: User identifier
            message: Reminder message
            scheduled_time: When to send the reminder
            phone_number: User's WhatsApp phone number
            
        Returns:
            Job ID
        """
        if not self._started:
            await self.initialize()
        
        try:
            job_id = f"reminder_{reminder_id}"
            
            # Ensure scheduled_time is timezone-aware
            if scheduled_time.tzinfo is None:
                scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
            
            # Create job data
            job_data = {
                "reminder_id": reminder_id,
                "user_id": user_id,
                "message": message,
                "phone_number": phone_number,
                "scheduled_time": scheduled_time.isoformat(),
                "job_type": "reminder"
            }
            
            logger.info(f"Scheduling reminder for: {scheduled_time} (current time: {datetime.now(timezone.utc)})")
            
            # Schedule the job
            job = self.scheduler.add_job(
                func=self.reminder_job.send_reminder,
                trigger=DateTrigger(run_date=scheduled_time),
                args=[job_data],
                id=job_id,
                name=f"Reminder: {message[:50]}...",
                misfire_grace_time=600,  # 10 minutes grace period
                replace_existing=True
            )
            
            # Store job info
            self.jobs[job_id] = {
                "job_id": job_id,
                "type": "reminder",
                "user_id": user_id,
                "scheduled_time": scheduled_time.isoformat(),
                "data": job_data,
                "created_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Scheduled reminder {reminder_id} for {scheduled_time}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to schedule reminder {reminder_id}: {e}")
            raise
    
    async def schedule_recurring_task(
        self,
        task_id: str,
        user_id: str,
        task_type: str,
        interval_hours: int = 24,
        start_time: Optional[datetime] = None
    ) -> str:
        """
        Schedule a recurring task
        
        Args:
            task_id: Unique task identifier
            user_id: User identifier
            task_type: Type of task (daily_summary, weekly_report, etc.)
            interval_hours: Interval in hours
            start_time: When to start (default: now + interval)
            
        Returns:
            Job ID
        """
        if not self._started:
            await self.initialize()
        
        try:
            job_id = f"recurring_{task_id}"
            
            if not start_time:
                start_time = datetime.utcnow() + timedelta(hours=interval_hours)
            
            job_data = {
                "task_id": task_id,
                "user_id": user_id,
                "task_type": task_type,
                "interval_hours": interval_hours,
                "job_type": "recurring"
            }
            
            # Schedule recurring job
            job = self.scheduler.add_job(
                func=self.reminder_job.execute_recurring_task,
                trigger=IntervalTrigger(hours=interval_hours, start_date=start_time),
                args=[job_data],
                id=job_id,
                name=f"Recurring: {task_type}",
                replace_existing=True
            )
            
            # Store job info
            self.jobs[job_id] = {
                "job_id": job_id,
                "type": "recurring",
                "user_id": user_id,
                "task_type": task_type,
                "interval_hours": interval_hours,
                "start_time": start_time.isoformat(),
                "data": job_data,
                "created_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Scheduled recurring task {task_id} every {interval_hours} hours")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to schedule recurring task {task_id}: {e}")
            raise
    
    async def schedule_daily_summary(
        self,
        user_id: str,
        phone_number: str,
        time_hour: int = 18,  # 6 PM
        time_minute: int = 0
    ) -> str:
        """Schedule daily summary for user"""
        job_id = f"daily_summary_{user_id}"
        
        job_data = {
            "user_id": user_id,
            "phone_number": phone_number,
            "job_type": "daily_summary"
        }
        
        # Schedule daily at specified time
        job = self.scheduler.add_job(
            func=self.reminder_job.send_daily_summary,
            trigger=CronTrigger(hour=time_hour, minute=time_minute),
            args=[job_data],
            id=job_id,
            name=f"Daily Summary for {user_id}",
            replace_existing=True
        )
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "type": "daily_summary",
            "user_id": user_id,
            "time": f"{time_hour:02d}:{time_minute:02d}",
            "data": job_data,
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Scheduled daily summary for {user_id} at {time_hour:02d}:{time_minute:02d}")
        return job_id
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        try:
            if job_id in self.jobs:
                self.scheduler.remove_job(job_id)
                del self.jobs[job_id]
                logger.info(f"Cancelled job {job_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def get_user_jobs(self, user_id: str) -> list[Dict[str, Any]]:
        """Get all jobs for a user"""
        return [job for job in self.jobs.values() if job.get("user_id") == user_id]
    
    async def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific job"""
        return self.jobs.get(job_id)
    
    async def reschedule_reminder(
        self,
        reminder_id: str,
        new_time: datetime
    ) -> bool:
        """Reschedule an existing reminder"""
        try:
            job_id = f"reminder_{reminder_id}"
            
            if job_id not in self.jobs:
                return False
            
            job_info = self.jobs[job_id]
            job_data = job_info["data"]
            job_data["scheduled_time"] = new_time.isoformat()
            
            # Remove old job and create new one
            self.scheduler.remove_job(job_id)
            
            job = self.scheduler.add_job(
                func=self.reminder_job.send_reminder,
                trigger=DateTrigger(run_date=new_time),
                args=[job_data],
                id=job_id,
                name=f"Reminder: {job_data['message'][:50]}...",
                misfire_grace_time=600,
                replace_existing=True
            )
            
            # Update job info
            job_info["scheduled_time"] = new_time.isoformat()
            job_info["updated_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Rescheduled reminder {reminder_id} to {new_time}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reschedule reminder {reminder_id}: {e}")
            return False
    
    async def get_next_jobs(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Get next scheduled jobs"""
        try:
            jobs = self.scheduler.get_jobs()
            next_jobs = []
            
            for job in jobs[:limit]:
                job_info = self.jobs.get(job.id, {})
                next_jobs.append({
                    "job_id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "user_id": job_info.get("user_id"),
                    "type": job_info.get("type"),
                    "data": job_info.get("data", {})
                })
            
            return next_jobs
            
        except Exception as e:
            logger.error(f"Failed to get next jobs: {e}")
            return []
    
    async def shutdown(self):
        """Shutdown the scheduler"""
        if self.scheduler and self._started:
            self.scheduler.shutdown()
            self._started = False
            logger.info("Scheduler shutdown completed")
    
    def _job_executed(self, event):
        """Handle job execution events"""
        logger.info(f"✅ Job executed successfully: {event.job_id}")
        if event.retval:
            logger.info(f"Job return value: {event.retval}")
    
    def _job_error(self, event):
        """Handle job error events"""
        logger.error(f"❌ Job error: {event.job_id} - {event.exception}")
        logger.error(f"Exception details: {event.traceback}")
    
    def _job_missed(self, event):
        """Handle missed job events"""
        logger.warning(f"⚠️ Job missed: {event.job_id} - scheduled time passed")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        if not self._started:
            return {"status": "not_started"}
        
        try:
            jobs = self.scheduler.get_jobs()
            
            stats = {
                "status": "running",
                "total_jobs": len(jobs),
                "job_types": {},
                "users": set(),
                "next_run": None
            }
            
            for job in jobs:
                job_info = self.jobs.get(job.id, {})
                job_type = job_info.get("type", "unknown")
                user_id = job_info.get("user_id")
                
                stats["job_types"][job_type] = stats["job_types"].get(job_type, 0) + 1
                
                if user_id:
                    stats["users"].add(user_id)
                
                if job.next_run_time:
                    if not stats["next_run"] or job.next_run_time < stats["next_run"]:
                        stats["next_run"] = job.next_run_time.isoformat()
            
            stats["unique_users"] = len(stats["users"])
            stats["users"] = list(stats["users"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get scheduler stats: {e}")
            return {"status": "error", "error": str(e)}

# Global scheduler instance
scheduler = SecondBrainScheduler()

# Test function
async def test_scheduler():
    """Test scheduler functionality"""
    
    print("🧪 Testing Scheduler")
    print("=" * 50)
    
    try:
        # Initialize scheduler
        await scheduler.initialize()
        print("✅ Scheduler initialized")
        
        # Schedule a test reminder
        test_time = datetime.utcnow() + timedelta(seconds=30)
        job_id = await scheduler.schedule_reminder(
            reminder_id="test_001",
            user_id="test_user",
            message="Test reminder message",
            scheduled_time=test_time,
            phone_number="whatsapp:+1234567890"
        )
        print(f"✅ Scheduled test reminder: {job_id}")
        
        # Get job info
        job_info = await scheduler.get_job_info(job_id)
        print(f"✅ Job info: {job_info}")
        
        # Get stats
        stats = await scheduler.get_stats()
        print(f"✅ Scheduler stats: {stats}")
        
        # Cancel the test job
        cancelled = await scheduler.cancel_job(job_id)
        print(f"✅ Job cancelled: {cancelled}")
        
        print("✅ All scheduler tests passed!")
        
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_scheduler())
