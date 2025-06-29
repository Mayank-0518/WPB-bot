"""
Scheduler Jobs for WhatsApp AI Second Brain Assistant
Contains job definitions for reminders, summaries, and periodic tasks

Example Usage:
# Send reminder job
await reminder_job.send_reminder({
    "reminder_id": "rem_001",
    "user_id": "user123",
    "message": "Don't forget to review the quarterly report",
    "phone_number": "whatsapp:+1234567890"
})

# Daily summary job
await reminder_job.send_daily_summary({
    "user_id": "user123", 
    "phone_number": "whatsapp:+1234567890"
})
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from twilio.rest import Client
from backend.utils.config import config
from backend.ai.summarizer import summarizer
from backend.memory.vectorstore import vector_store

logger = logging.getLogger(__name__)

class ReminderJob:
    """Job handler for reminders and periodic tasks"""
    
    def __init__(self):
        self.twilio_client = None
        self._init_twilio()
    
    def _init_twilio(self):
        """Initialize Twilio client"""
        try:
            self.twilio_client = Client(
                config.TWILIO_ACCOUNT_SID,
                config.TWILIO_AUTH_TOKEN
            )
            logger.info("Twilio client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
    
    async def send_reminder(self, job_data: Dict[str, Any]):
        """
        Send a WhatsApp reminder message
        
        Args:
            job_data: Dictionary containing reminder details
                - reminder_id: Unique reminder ID
                - user_id: User identifier
                - message: Reminder message
                - phone_number: WhatsApp phone number
        """
        try:
            logger.info(f"🔔 Executing reminder job with data: {job_data}")
            
            reminder_id = job_data.get("reminder_id")
            user_id = job_data.get("user_id")
            message = job_data.get("message")
            phone_number = job_data.get("phone_number")
            
            if not all([reminder_id, user_id, message, phone_number]):
                logger.error(f"Missing required reminder data: {job_data}")
                return
            
            # Format reminder message
            reminder_text = f"🔔 Reminder: {message}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            logger.info(f"Sending reminder to {phone_number}: {reminder_text[:100]}...")
            
            # Send WhatsApp message
            await self._send_whatsapp_message(phone_number, reminder_text)
            
            logger.info(f"✅ Successfully sent reminder {reminder_id} to {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send reminder: {e}")
            raise
    
    async def send_daily_summary(self, job_data: Dict[str, Any]):
        """
        Send daily summary to user
        
        Args:
            job_data: Dictionary containing user details
                - user_id: User identifier
                - phone_number: WhatsApp phone number
        """
        try:
            user_id = job_data.get("user_id")
            phone_number = job_data.get("phone_number")
            
            if not all([user_id, phone_number]):
                logger.error(f"Missing required summary data: {job_data}")
                return
            
            # Generate daily summary
            summary_text = await self._generate_daily_summary(user_id)
            
            # Send WhatsApp message
            await self._send_whatsapp_message(phone_number, summary_text)
            
            logger.info(f"Sent daily summary to {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
    
    async def execute_recurring_task(self, job_data: Dict[str, Any]):
        """
        Execute recurring tasks
        
        Args:
            job_data: Dictionary containing task details
                - task_id: Task identifier
                - user_id: User identifier
                - task_type: Type of recurring task
        """
        try:
            task_id = job_data.get("task_id")
            user_id = job_data.get("user_id")
            task_type = job_data.get("task_type")
            
            logger.info(f"Executing recurring task {task_id} of type {task_type} for {user_id}")
            
            if task_type == "daily_summary":
                await self.send_daily_summary(job_data)
            elif task_type == "weekly_report":
                await self._send_weekly_report(job_data)
            elif task_type == "backup_reminder":
                await self._send_backup_reminder(job_data)
            else:
                logger.warning(f"Unknown recurring task type: {task_type}")
            
        except Exception as e:
            logger.error(f"Failed to execute recurring task: {e}")
    
    async def _generate_daily_summary(self, user_id: str) -> str:
        """Generate daily summary for user"""
        try:
            # Initialize vector store if needed
            if not vector_store._initialized:
                await vector_store.initialize()
            
            # Get today's documents (simplified - in practice, filter by date)
            user_docs = await vector_store.get_user_documents(user_id)
            
            if not user_docs:
                return "📊 Daily Summary\n\nNo new content added today. Consider adding some notes, documents, or tasks to build your second brain!"
            
            # Get recent documents (last 5)
            recent_docs = sorted(user_docs, key=lambda x: x.get("created_at", ""), reverse=True)[:5]
            
            # Create summary content
            summary_parts = ["📊 Daily Summary\n"]
            
            # Document count
            summary_parts.append(f"📄 Documents: {len(user_docs)} total")
            
            # Recent activity
            if recent_docs:
                summary_parts.append("\n🔄 Recent Activity:")
                for i, doc in enumerate(recent_docs[:3], 1):
                    title = doc.get("metadata", {}).get("title", "Untitled")
                    content_preview = doc.get("content", "")[:50]
                    summary_parts.append(f"{i}. {title}: {content_preview}...")
            
            # Quick stats
            total_words = sum(len(doc.get("content", "").split()) for doc in user_docs)
            summary_parts.append(f"\n📈 Total Content: {total_words:,} words")
            
            # Tips or suggestions
            summary_parts.append("\n💡 Tip: Ask me questions about your documents or set reminders for important tasks!")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate daily summary for {user_id}: {e}")
            return "📊 Daily Summary\n\nSorry, I couldn't generate your summary right now. Please try again later."
    
    async def _send_weekly_report(self, job_data: Dict[str, Any]):
        """Send weekly report"""
        try:
            user_id = job_data.get("user_id")
            phone_number = job_data.get("phone_number")
            
            report_text = f"""📈 Weekly Report for {user_id}

📊 This Week's Activity:
- Documents processed: X
- Questions answered: Y
- Reminders sent: Z

🎯 Top Topics:
1. Topic A
2. Topic B
3. Topic C

💡 Suggestion: Consider organizing your notes by topic for better retrieval.

Have a great week ahead! 🚀"""
            
            await self._send_whatsapp_message(phone_number, report_text)
            
        except Exception as e:
            logger.error(f"Failed to send weekly report: {e}")
    
    async def _send_backup_reminder(self, job_data: Dict[str, Any]):
        """Send backup reminder"""
        try:
            phone_number = job_data.get("phone_number")
            
            backup_text = """🔄 Backup Reminder

Don't forget to backup your important documents and data!

Your AI Second Brain is working hard to remember everything for you, but it's always good to have backups of critical information.

Stay organized! 📚"""
            
            await self._send_whatsapp_message(phone_number, backup_text)
            
        except Exception as e:
            logger.error(f"Failed to send backup reminder: {e}")
    
    async def _send_whatsapp_message(self, phone_number: str, message: str):
        """Send WhatsApp message via Twilio"""
        try:
            if not self.twilio_client:
                self._init_twilio()
            
            if not self.twilio_client:
                logger.error("Twilio client not available")
                return
            
            # Ensure phone number format
            if not phone_number.startswith("whatsapp:"):
                phone_number = f"whatsapp:{phone_number}"
            
            # Send message - run in thread pool since Twilio client is synchronous
            loop = asyncio.get_event_loop()
            message_instance = await loop.run_in_executor(
                None,
                lambda: self.twilio_client.messages.create(
                    body=message,
                    from_=config.TWILIO_PHONE_NUMBER,
                    to=phone_number
                )
            )
            
            logger.info(f"WhatsApp reminder message sent: {message_instance.sid}")
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            # Re-raise to ensure job failure is logged
            raise
    
    async def send_task_reminder(self, job_data: Dict[str, Any]):
        """Send task-specific reminder"""
        try:
            task_title = job_data.get("task_title", "Task")
            due_date = job_data.get("due_date")
            phone_number = job_data.get("phone_number")
            
            if due_date:
                due_text = f"\n⏰ Due: {due_date}"
            else:
                due_text = ""
            
            task_message = f"""📋 Task Reminder: {task_title}{due_text}

Don't forget to complete this task. Reply with 'done' when finished, or 'snooze 1h' to remind you again later.

Stay productive! ✅"""
            
            await self._send_whatsapp_message(phone_number, task_message)
            
        except Exception as e:
            logger.error(f"Failed to send task reminder: {e}")
    
    async def send_follow_up_reminder(self, job_data: Dict[str, Any]):
        """Send follow-up reminder for unanswered questions"""
        try:
            question = job_data.get("question", "Previous question")
            phone_number = job_data.get("phone_number")
            
            follow_up_message = f"""❓ Follow-up Reminder

You asked: "{question}"

I haven't received a response yet. If you need more information or have additional questions, feel free to ask!

I'm here to help! 🤖"""
            
            await self._send_whatsapp_message(phone_number, follow_up_message)
            
        except Exception as e:
            logger.error(f"Failed to send follow-up reminder: {e}")
    
    async def health_check(self):
        """Health check for the job system"""
        try:
            # Check Twilio connection
            if self.twilio_client:
                # Try to get account info (minimal API call)
                account = self.twilio_client.api.accounts(config.TWILIO_ACCOUNT_SID).fetch()
                logger.info(f"Twilio health check passed: {account.status}")
                return True
            else:
                logger.warning("Twilio client not initialized")
                return False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Global job handler instance
reminder_job = ReminderJob()

# Test function
async def test_reminder_jobs():
    """Test reminder job functionality"""
    
    print("🧪 Testing Reminder Jobs")
    print("=" * 50)
    
    try:
        # Test Twilio health check
        health_ok = await reminder_job.health_check()
        print(f"✅ Twilio health check: {'OK' if health_ok else 'Failed'}")
        
        # Test daily summary generation
        summary = await reminder_job._generate_daily_summary("test_user")
        print(f"✅ Daily summary generated: {len(summary)} characters")
        print(f"   Preview: {summary[:100]}...")
        
        print("✅ All reminder job tests completed!")
        
    except Exception as e:
        print(f"❌ Reminder job test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_reminder_jobs())
