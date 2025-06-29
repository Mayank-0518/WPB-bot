"""
Reminders API Routes for WhatsApp AI Second Brain Assistant
Provides REST API endpoints for managing reminders and scheduled tasks

Example Usage:
POST /reminders/
{
    "user_id": "user123",
    "title": "Review quarterly report",
    "message": "Don't forget to review the Q2 report",
    "scheduled_time": "2025-06-30T14:00:00Z",
    "phone_number": "whatsapp:+1234567890"
}

GET /reminders/user123
[
    {
        "reminder_id": "rem_001",
        "title": "Review quarterly report", 
        "scheduled_time": "2025-06-30T14:00:00Z",
        "status": "pending"
    }
]
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.models.schema import ReminderItem, ReminderStatus
from backend.scheduler.scheduler import scheduler
from backend.utils.time_parser import parse_time_expression

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/reminders", tags=["reminders"])

class CreateReminderRequest(BaseModel):
    """Request model for creating reminders"""
    user_id: str
    title: str
    message: str
    scheduled_time: Optional[str] = None  # ISO string or natural language
    phone_number: str
    category: str = "general"

class UpdateReminderRequest(BaseModel):
    """Request model for updating reminders"""
    title: Optional[str] = None
    message: Optional[str] = None
    scheduled_time: Optional[str] = None
    status: Optional[ReminderStatus] = None

class ReminderResponse(BaseModel):
    """Response model for reminder data"""
    reminder_id: str
    user_id: str
    title: str
    message: str
    scheduled_time: str
    status: str
    created_at: str
    job_id: Optional[str] = None

class ReminderManager:
    """Manager for reminder operations"""
    
    def __init__(self):
        self.reminders_db = {}  # In-memory storage (use real DB in production)
    
    async def create_reminder(self, request: CreateReminderRequest) -> ReminderResponse:
        """Create a new reminder"""
        try:
            # Parse scheduled time
            if request.scheduled_time:
                try:
                    # Try parsing as ISO format first
                    scheduled_dt = datetime.fromisoformat(request.scheduled_time.replace("Z", "+00:00"))
                except ValueError:
                    # Try natural language parsing
                    scheduled_dt = parse_time_expression(request.scheduled_time)
                    if not scheduled_dt:
                        raise ValueError("Could not parse scheduled time")
            else:
                # Default to 1 hour from now
                scheduled_dt = datetime.utcnow() + timedelta(hours=1)
            
            # Generate reminder ID
            import uuid
            reminder_id = str(uuid.uuid4())
            
            # Schedule the reminder
            job_id = await scheduler.schedule_reminder(
                reminder_id=reminder_id,
                user_id=request.user_id,
                message=request.message,
                scheduled_time=scheduled_dt,
                phone_number=request.phone_number
            )
            
            # Create reminder record
            reminder = ReminderResponse(
                reminder_id=reminder_id,
                user_id=request.user_id,
                title=request.title,
                message=request.message,
                scheduled_time=scheduled_dt.isoformat(),
                status=ReminderStatus.PENDING.value,
                created_at=datetime.utcnow().isoformat(),
                job_id=job_id
            )
            
            # Store in memory (use real DB in production)
            self.reminders_db[reminder_id] = reminder
            
            logger.info(f"Created reminder {reminder_id} for user {request.user_id}")
            return reminder
            
        except Exception as e:
            logger.error(f"Failed to create reminder: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    async def get_user_reminders(
        self, 
        user_id: str, 
        status: Optional[ReminderStatus] = None
    ) -> List[ReminderResponse]:
        """Get all reminders for a user"""
        try:
            user_reminders = [
                reminder for reminder in self.reminders_db.values()
                if reminder.user_id == user_id
            ]
            
            # Filter by status if specified
            if status:
                user_reminders = [
                    reminder for reminder in user_reminders
                    if reminder.status == status.value
                ]
            
            # Sort by scheduled time
            user_reminders.sort(key=lambda x: x.scheduled_time)
            
            return user_reminders
            
        except Exception as e:
            logger.error(f"Failed to get user reminders: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_reminder(self, reminder_id: str) -> ReminderResponse:
        """Get a specific reminder"""
        if reminder_id not in self.reminders_db:
            raise HTTPException(status_code=404, detail="Reminder not found")
        
        return self.reminders_db[reminder_id]
    
    async def update_reminder(
        self, 
        reminder_id: str, 
        request: UpdateReminderRequest
    ) -> ReminderResponse:
        """Update an existing reminder"""
        try:
            if reminder_id not in self.reminders_db:
                raise HTTPException(status_code=404, detail="Reminder not found")
            
            reminder = self.reminders_db[reminder_id]
            
            # Update fields
            if request.title:
                reminder.title = request.title
            if request.message:
                reminder.message = request.message  
            if request.status:
                reminder.status = request.status.value
            
            # Update scheduled time if provided
            if request.scheduled_time:
                try:
                    scheduled_dt = datetime.fromisoformat(request.scheduled_time.replace("Z", "+00:00"))
                except ValueError:
                    scheduled_dt = parse_time_expression(request.scheduled_time)
                    if not scheduled_dt:
                        raise ValueError("Could not parse scheduled time")
                
                reminder.scheduled_time = scheduled_dt.isoformat()
                
                # Reschedule the job
                await scheduler.reschedule_reminder(reminder_id, scheduled_dt)
            
            logger.info(f"Updated reminder {reminder_id}")
            return reminder
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update reminder: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    async def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder"""
        try:
            if reminder_id not in self.reminders_db:
                raise HTTPException(status_code=404, detail="Reminder not found")
            
            # Cancel scheduled job
            job_id = f"reminder_{reminder_id}"
            await scheduler.cancel_job(job_id)
            
            # Remove from storage
            del self.reminders_db[reminder_id]
            
            logger.info(f"Deleted reminder {reminder_id}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete reminder: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_upcoming_reminders(
        self, 
        user_id: str, 
        hours_ahead: int = 24
    ) -> List[ReminderResponse]:
        """Get upcoming reminders for a user"""
        try:
            cutoff_time = datetime.utcnow() + timedelta(hours=hours_ahead)
            
            upcoming = []
            for reminder in self.reminders_db.values():
                if (reminder.user_id == user_id and 
                    reminder.status == ReminderStatus.PENDING.value):
                    
                    reminder_time = datetime.fromisoformat(reminder.scheduled_time.replace("Z", "+00:00"))
                    if datetime.utcnow() <= reminder_time <= cutoff_time:
                        upcoming.append(reminder)
            
            # Sort by scheduled time
            upcoming.sort(key=lambda x: x.scheduled_time)
            return upcoming
            
        except Exception as e:
            logger.error(f"Failed to get upcoming reminders: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global reminder manager
reminder_manager = ReminderManager()

@router.post("/", response_model=ReminderResponse)
async def create_reminder(request: CreateReminderRequest):
    """Create a new reminder"""
    return await reminder_manager.create_reminder(request)

@router.get("/{user_id}", response_model=List[ReminderResponse])
async def get_user_reminders(
    user_id: str,
    status: Optional[ReminderStatus] = Query(None, description="Filter by status")
):
    """Get all reminders for a user"""
    return await reminder_manager.get_user_reminders(user_id, status)

@router.get("/reminder/{reminder_id}", response_model=ReminderResponse)
async def get_reminder(reminder_id: str):
    """Get a specific reminder"""
    return await reminder_manager.get_reminder(reminder_id)

@router.put("/{reminder_id}", response_model=ReminderResponse)
async def update_reminder(
    reminder_id: str,
    request: UpdateReminderRequest
):
    """Update an existing reminder"""
    return await reminder_manager.update_reminder(reminder_id, request)

@router.delete("/{reminder_id}")
async def delete_reminder(reminder_id: str):
    """Delete a reminder"""
    success = await reminder_manager.delete_reminder(reminder_id)
    return {"success": success, "reminder_id": reminder_id}

@router.get("/{user_id}/upcoming", response_model=List[ReminderResponse])
async def get_upcoming_reminders(
    user_id: str,
    hours: int = Query(24, description="Hours to look ahead", ge=1, le=168)
):
    """Get upcoming reminders for a user"""
    return await reminder_manager.get_upcoming_reminders(user_id, hours)

@router.post("/{user_id}/daily-summary")
async def enable_daily_summary(
    user_id: str,
    phone_number: str,
    time_hour: int = Query(18, description="Hour to send summary (0-23)", ge=0, le=23),
    time_minute: int = Query(0, description="Minute to send summary (0-59)", ge=0, le=59)
):
    """Enable daily summary for a user"""
    try:
        job_id = await scheduler.schedule_daily_summary(
            user_id=user_id,
            phone_number=phone_number,
            time_hour=time_hour,
            time_minute=time_minute
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "summary_time": f"{time_hour:02d}:{time_minute:02d}",
            "message": f"Daily summary enabled for {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to enable daily summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{user_id}/daily-summary")
async def disable_daily_summary(user_id: str):
    """Disable daily summary for a user"""
    try:
        job_id = f"daily_summary_{user_id}"
        success = await scheduler.cancel_job(job_id)
        
        return {
            "success": success,
            "message": f"Daily summary disabled for {user_id}" if success else "Daily summary was not enabled"
        }
        
    except Exception as e:
        logger.error(f"Failed to disable daily summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{user_id}")
async def get_reminder_stats(user_id: str):
    """Get reminder statistics for a user"""
    try:
        all_reminders = await reminder_manager.get_user_reminders(user_id)
        
        stats = {
            "total_reminders": len(all_reminders),
            "pending": len([r for r in all_reminders if r.status == ReminderStatus.PENDING.value]),
            "completed": len([r for r in all_reminders if r.status == ReminderStatus.COMPLETED.value]),
            "cancelled": len([r for r in all_reminders if r.status == ReminderStatus.CANCELLED.value])
        }
        
        # Get upcoming reminders
        upcoming = await reminder_manager.get_upcoming_reminders(user_id, 24)
        stats["upcoming_24h"] = len(upcoming)
        
        # Next reminder
        if upcoming:
            stats["next_reminder"] = {
                "title": upcoming[0].title,
                "scheduled_time": upcoming[0].scheduled_time
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get reminder stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
