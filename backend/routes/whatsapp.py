"""
WhatsApp Route Handler for AI Second Brain Assistant
Processes incoming WhatsApp messages and routes them to appropriate handlers

Example Flow:
1. Twilio webhook receives WhatsApp message
2. Message is parsed and user intent is determined
3. Based on intent, message is routed to:
   - Document summarization
   - Task extraction  
   - Question answering
   - Reminder setting
   - General conversation

Example webhook payload from Twilio:
{
    "From": "whatsapp:+1234567890",
    "To": "whatsapp:+14155238886", 
    "Body": "Please summarize this document and remind me to review it tomorrow",
    "MediaUrl0": "https://api.twilio.com/media/document.pdf"
}
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from backend.utils.config import config
from backend.models.schema import WhatsAppMessage, MessageType
from backend.ai.granite_api import granite_client
from backend.ai.summarizer import summarizer  
from backend.ai.task_extractor import task_extractor
from backend.ai.qa import qa_system
from backend.memory.vectorstore import vector_store
from backend.files.parser import file_parser
from backend.scheduler.scheduler import scheduler
from backend.utils.time_parser import parse_time_expression

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class WhatsAppHandler:
    """Main handler for WhatsApp messages"""
    
    def __init__(self):
        self.twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        self.user_sessions = {}  # Store user conversation context
        
    async def process_message(self, message_data: Dict[str, Any]) -> str:
        """
        Process incoming WhatsApp message and generate response
        
        Args:
            message_data: Twilio webhook data
            
        Returns:
            Response message to send back
        """
        try:
            # Parse message
            whatsapp_msg = self._parse_twilio_message(message_data)
            
            # Get or create user session
            user_id = self._get_user_id(whatsapp_msg.from_number)
            
            # Initialize services if needed
            await self._ensure_services_initialized()
            
            # Determine message intent
            intent = await self._determine_intent(whatsapp_msg.body)
            
            logger.info(f"Processing message from {user_id} with intent: {intent}")
            
            # Route to appropriate handler
            if intent == "summarize":
                return await self._handle_summarize_request(whatsapp_msg, user_id)
            elif intent == "question":
                return await self._handle_question(whatsapp_msg, user_id)
            elif intent == "reminder":
                return await self._handle_reminder_request(whatsapp_msg, user_id)
            elif intent == "task":
                return await self._handle_task_extraction(whatsapp_msg, user_id)
            elif intent == "note":
                return await self._handle_note_storage(whatsapp_msg, user_id)
            else:
                return await self._handle_general_conversation(whatsapp_msg, user_id)
                
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return "Sorry, I encountered an error processing your message. Please try again."
    
    def _parse_twilio_message(self, data: Dict[str, Any]) -> WhatsAppMessage:
        """Parse Twilio webhook data into WhatsAppMessage"""
        return WhatsAppMessage(
            from_number=data.get("From", ""),
            to_number=data.get("To", ""),
            body=data.get("Body", ""),
            media_url=data.get("MediaUrl0"),  # First media attachment
            message_type=MessageType.DOCUMENT if data.get("MediaUrl0") else MessageType.TEXT
        )
    
    def _get_user_id(self, phone_number: str) -> str:
        """Extract user ID from phone number"""
        # Remove 'whatsapp:' prefix and clean phone number
        clean_number = phone_number.replace("whatsapp:", "").replace("+", "").replace("-", "").replace(" ", "")
        return f"user_{clean_number}"
    
    async def _ensure_services_initialized(self):
        """Ensure all AI services are initialized"""
        if not vector_store._initialized:
            await vector_store.initialize()
        if not scheduler._started:
            await scheduler.initialize()
    
    async def _determine_intent(self, message_body: str) -> str:
        """Determine user intent from message"""
        try:
            intent = await granite_client.categorize_message(message_body)
            return intent
        except Exception as e:
            logger.warning(f"Intent determination failed: {e}")
            # Fallback to keyword-based intent detection
            return self._fallback_intent_detection(message_body)
    
    def _fallback_intent_detection(self, message: str) -> str:
        """Fallback intent detection using keywords"""
        message_lower = message.lower()
        
        # Summarization keywords
        if any(word in message_lower for word in ["summarize", "summary", "tldr", "brief"]):
            return "summarize"
        
        # Question keywords  
        if any(word in message_lower for word in ["what", "how", "why", "when", "where", "?"]):
            return "question"
        
        # Reminder keywords
        if any(word in message_lower for word in ["remind", "reminder", "don't forget", "alert"]):
            return "reminder"
        
        # Task keywords
        if any(word in message_lower for word in ["task", "todo", "need to", "should", "must"]):
            return "task"
        
        # Note storage keywords
        if any(word in message_lower for word in ["note", "save", "remember", "store"]):
            return "note"
        
        return "general"
    
    async def _handle_summarize_request(self, message: WhatsAppMessage, user_id: str) -> str:
        """Handle document summarization requests"""
        try:
            content_to_summarize = ""
            
            # Check if there's a media attachment
            if message.media_url:
                # Download and parse the file
                try:
                    file_content = await self._download_and_parse_media(message.media_url)
                    # Check if the result is an error message
                    if file_content.startswith("Error"):
                        return f"Sorry, I couldn't process the uploaded file: {file_content}"
                    content_to_summarize = file_content
                except Exception as e:
                    return f"Sorry, I couldn't process the uploaded file: {str(e)}"
            else:
                # Use the message body as content
                content_to_summarize = message.body
            
            if not content_to_summarize or len(content_to_summarize.strip()) < 50:
                return "Please provide content to summarize, either as text or by uploading a document."
            
            # Generate summary
            summary_result = await summarizer.summarize_document(
                content_to_summarize,
                summary_type="concise",
                max_length=150
            )
            
            # Store the original content and summary
            await vector_store.add_document(
                user_id=user_id,
                content=content_to_summarize,
                metadata={
                    "type": "summarized_document",
                    "summary": summary_result.summary,
                    "source": "whatsapp",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Format response
            response = f"📄 Summary:\n{summary_result.summary}\n\n"
            
            if summary_result.key_points:
                response += "🔑 Key Points:\n"
                for i, point in enumerate(summary_result.key_points, 1):
                    response += f"{i}. {point}\n"
            
            response += f"\n📊 Word count: {summary_result.word_count}"
            
            return response
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Sorry, I couldn't summarize the content. Please check the format and try again."
    
    async def _handle_question(self, message: WhatsAppMessage, user_id: str) -> str:
        """Handle Q&A requests"""
        try:
            question = message.body
            
            # Get answer using RAG
            qa_result = await qa_system.answer_question(question, user_id)
            
            # Format response  
            response = f"💡 Answer:\n{qa_result.answer}"
            
            if qa_result.sources:
                response += f"\n\n📚 Sources: {', '.join(qa_result.sources[:3])}"
            
            if qa_result.confidence < 0.3:
                response += "\n\n⚠️ Note: I'm not very confident about this answer. Please verify the information."
            
            return response
            
        except Exception as e:
            logger.error(f"Q&A failed: {e}")
            return "Sorry, I couldn't answer your question right now. Please try rephrasing or provide more context."
    
    async def _handle_reminder_request(self, message: WhatsAppMessage, user_id: str) -> str:
        """Handle reminder setting requests"""
        try:
            # First try simple time parsing for common patterns
            reminder_time = self._parse_simple_time(message.body)
            
            if reminder_time:
                # Simple pattern found, use it directly
                from datetime import timezone
                current_time = datetime.now(timezone.utc)
                scheduled_time = current_time + reminder_time
                
                # Extract message content (remove time expressions)
                reminder_text = self._extract_reminder_message(message.body)
                
                # Schedule reminder directly
                reminder_id = f"simple_{user_id}_{int(current_time.timestamp())}"
                job_id = await scheduler.schedule_reminder(
                    reminder_id=reminder_id,
                    user_id=user_id,
                    message=reminder_text,
                    scheduled_time=scheduled_time,
                    phone_number=message.from_number
                )
                
                return f"🔔 Reminder set!\n📝 {reminder_text}\n⏰ Scheduled for: {scheduled_time.strftime('%H:%M:%S')} UTC\n✅ Job ID: {job_id}"
            
            # Fallback to AI extraction
            extraction_result = await task_extractor.extract_tasks_and_reminders(message.body, user_id)
            
            reminders = extraction_result.get("reminders", [])
            
            if not reminders:
                return "I couldn't find any reminder requests in your message. Please try something like 'Remind me to call John in 30 seconds'."
            
            scheduled_count = 0
            response_parts = ["📋 Reminders set:"]
            
            for reminder in reminders[:3]:  # Limit to 3 reminders per message
                try:
                    scheduled_time_str = reminder.get("scheduled_time")
                    if scheduled_time_str:
                        scheduled_time = datetime.fromisoformat(scheduled_time_str.replace("Z", "+00:00"))
                    else:
                        # Default to 1 hour from now (timezone-aware)
                        from datetime import timezone
                        scheduled_time = datetime.now(timezone.utc) + timedelta(hours=1)
                    
                    # Schedule reminder
                    job_id = await scheduler.schedule_reminder(
                        reminder_id=reminder["reminder_id"],
                        user_id=user_id,
                        message=reminder["message"],
                        scheduled_time=scheduled_time,
                        phone_number=message.from_number
                    )
                    
                    scheduled_count += 1
                    response_parts.append(f"• {reminder['title']} - {scheduled_time.strftime('%Y-%m-%d %H:%M')}")
                    
                except Exception as e:
                    logger.warning(f"Failed to schedule reminder: {e}")
                    continue
            
            if scheduled_count == 0:
                return "Sorry, I couldn't schedule any reminders. Please check the time format and try again."
            
            response_parts.append(f"\n✅ Successfully scheduled {scheduled_count} reminder(s)!")
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Reminder handling failed: {e}")
            return "Sorry, I couldn't process your reminder request. Please try again with a clearer format."
    
    def _parse_simple_time(self, text: str) -> Optional[timedelta]:
        """Parse simple time expressions like 'in 30 seconds', 'in 2 minutes'"""
        import re
        text_lower = text.lower()
        
        # Pattern for "in X seconds"
        seconds_match = re.search(r'in (\d+) seconds?', text_lower)
        if seconds_match:
            return timedelta(seconds=int(seconds_match.group(1)))
        
        # Pattern for "in X minutes"
        minutes_match = re.search(r'in (\d+) minutes?', text_lower)
        if minutes_match:
            return timedelta(minutes=int(minutes_match.group(1)))
        
        # Pattern for "in X hours"
        hours_match = re.search(r'in (\d+) hours?', text_lower)
        if hours_match:
            return timedelta(hours=int(hours_match.group(1)))
        
        return None
    
    def _extract_reminder_message(self, text: str) -> str:
        """Extract the reminder message, removing time expressions"""
        import re
        
        # Remove common reminder prefixes
        text = re.sub(r'^remind me to\s*', '', text.lower())
        
        # Remove time expressions
        text = re.sub(r'\s*in \d+ (seconds?|minutes?|hours?)\s*', '', text.lower())
        
        # Clean up and capitalize
        text = text.strip()
        if text:
            return text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        else:
            return "Reminder"
    
    async def _handle_task_extraction(self, message: WhatsAppMessage, user_id: str) -> str:
        """Handle task extraction from messages"""
        try:
            # Extract tasks and reminders
            extraction_result = await task_extractor.extract_tasks_and_reminders(message.body, user_id)
            
            tasks = extraction_result.get("tasks", [])
            reminders = extraction_result.get("reminders", [])
            
            if not tasks and not reminders:
                # Store as general note instead
                return await self._handle_note_storage(message, user_id)
            
            response_parts = []
            
            if tasks:
                response_parts.append("📝 Tasks identified:")
                for task in tasks[:5]:  # Limit to 5 tasks
                    due_info = f" (Due: {task.get('due_date', 'TBD')})" if task.get('due_date') else ""
                    response_parts.append(f"• {task['title']}{due_info}")
            
            if reminders:
                response_parts.append("\n🔔 Reminders to schedule:")
                for reminder in reminders[:3]:  # Limit to 3 reminders
                    time_info = f" ({reminder.get('scheduled_time', 'TBD')})" if reminder.get('scheduled_time') else ""
                    response_parts.append(f"• {reminder['title']}{time_info}")
            
            # Store the original message
            await vector_store.add_document(
                user_id=user_id,
                content=message.body,
                metadata={
                    "type": "task_message",
                    "tasks_count": len(tasks),
                    "reminders_count": len(reminders),
                    "source": "whatsapp",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            response_parts.append("\n💾 Content saved to your second brain!")
            response_parts.append("\nSay 'set reminders' to schedule the reminders above.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Task extraction failed: {e}")
            return "Sorry, I couldn't extract tasks from your message. I've saved it as a note instead."
    
    async def _handle_note_storage(self, message: WhatsAppMessage, user_id: str) -> str:
        """Handle general note storage"""
        try:
            content = message.body
            
            # Generate title using AI
            title_prompt = f"Generate a short title (3-8 words) for this note: {content[:200]}"
            try:
                title = await granite_client.generate_text(title_prompt, max_tokens=20)
                title = title.strip().replace('"', '').replace("Title:", "").strip()
            except:
                title = f"Note from {datetime.now().strftime('%m/%d')}"
            
            # Store the note
            doc_id = await vector_store.add_document(
                user_id=user_id,
                content=content,
                metadata={
                    "type": "note",
                    "title": title,
                    "source": "whatsapp",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return f"📝 Note saved: '{title}'\n\nI've added this to your second brain. You can ask me questions about it later!"
            
        except Exception as e:
            logger.error(f"Note storage failed: {e}")
            return "Sorry, I couldn't save your note right now. Please try again."
    
    async def _handle_general_conversation(self, message: WhatsAppMessage, user_id: str) -> str:
        """Handle general conversation"""
        try:
            message_lower = message.body.lower()
            
            # Predefined responses for common requests to avoid AI hallucination
            if "translate" in message_lower and "french" in message_lower:
                if "good morning" in message_lower:
                    return "\"Good morning\" in French is \"Bonjour\"."
                elif "hello" in message_lower:
                    return "\"Hello\" in French is \"Bonjour\" or \"Salut\"."
                elif "thank you" in message_lower:
                    return "\"Thank you\" in French is \"Merci\"."
                else:
                    return "I can help with basic translations. What specific phrase would you like translated to French?"
            
            # Greeting responses
            if any(word in message_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
                return "Hello! I'm your AI Second Brain assistant. I can help you organize information, set reminders, answer questions, and summarize documents. How can I assist you today?"
            
            # Status requests
            if any(word in message_lower for word in ["status", "project", "ibm"]):
                return "I don't have access to external systems or real-time project data. Please check your project management tools or contact your team directly for current status updates."
            
            # Weather requests
            if "weather" in message_lower:
                return "I don't have access to current weather data. Please check a weather app or website for accurate forecasts."
            
            # Time/date requests
            if any(word in message_lower for word in ["time", "date", "today"]):
                return f"I can help you organize information and set reminders, but for current time/date, please check your device. I can set reminders for specific times if needed!"
            
            # Default fallback without using AI for general conversation
            user_docs = await vector_store.get_user_documents(user_id)
            
            response = "I understand you want to chat! I'm designed to help you with specific tasks like organizing information, setting reminders, and answering questions about your stored content."
            
            # Add helpful tips for new users
            if not user_docs:
                response += "\n\n💡 Tip: You can send me documents to summarize, ask questions, or say 'remind me to...' to set reminders!"
            
            return response
            
        except Exception as e:
            logger.error(f"General conversation failed: {e}")
            return """Hello! I'm your AI Second Brain assistant. I can help you:

📄 Summarize documents and content
❓ Answer questions about your stored information
🔔 Set reminders and extract tasks  
📝 Store and organize your notes

Try sending me a document, asking a question, or setting a reminder!"""
            
        except Exception as e:
            logger.error(f"General conversation failed: {e}")
            return """Hello! I'm your AI Second Brain assistant. I can help you:

📄 Summarize documents and content
❓ Answer questions about your stored information
🔔 Set reminders and extract tasks  
📝 Store and organize your notes

Try sending me a document, asking a question, or setting a reminder!"""
    
    async def _download_and_parse_media(self, media_url: str) -> str:
        """Download and parse media content"""
        try:
            import requests
            import tempfile
            import os
            from urllib.parse import urlparse
            
            logger.info(f"Downloading media from: {media_url}")
            
            # Handle local file URLs for testing
            if media_url.startswith('file://'):
                local_path = media_url.replace('file://', '').replace('file:', '')
                if os.path.exists(local_path):
                    logger.info(f"Processing local file: {local_path}")
                    try:
                        if local_path.lower().endswith('.pdf'):
                            # Try PDF parsing first
                            content = await file_parser.parse_pdf(local_path)
                        else:
                            # Text file
                            with open(local_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                    except Exception as pdf_error:
                        # If PDF parsing fails, try as text file
                        logger.warning(f"PDF parsing failed, trying as text: {pdf_error}")
                        try:
                            with open(local_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception as text_error:
                            return f"Error: Could not parse file as PDF or text: {text_error}"
                    return content
                else:
                    return f"Error: Local file not found: {local_path}"
            
            # Download media file with Twilio authentication for real URLs
            auth = (config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
            response = requests.get(media_url, auth=auth)
            
            if response.status_code != 200:
                logger.error(f"Failed to download media: {response.status_code}")
                return f"Error: Could not download media file (Status: {response.status_code})"
            
            # Get content type and file extension
            content_type = response.headers.get('content-type', '')
            logger.info(f"Media content type: {content_type}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Parse the file based on content type
                if 'text' in content_type:
                    # Text files
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif 'pdf' in content_type or media_url.lower().endswith('.pdf'):
                    # PDF files - use the file parser
                    try:
                        content = await file_parser.parse_pdf(temp_file_path)
                    except Exception as pdf_error:
                        # If PDF parsing fails, this might not be a valid PDF
                        logger.error(f"PDF parsing failed for Twilio media: {pdf_error}")
                        # Don't try to read binary PDF as text - just raise an error
                        raise Exception(f"Invalid or corrupted PDF file: {pdf_error}")
                elif any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/gif']):
                    # Image files - for now, just indicate it's an image
                    content = f"[Image file received - Content type: {content_type}]\nNote: Image text extraction not yet implemented."
                else:
                    # Unknown file type
                    content = f"[Unknown file type: {content_type}]\nFile size: {len(response.content)} bytes\nNote: This file type is not yet supported for text extraction."
                
                logger.info(f"Successfully parsed media content: {len(content)} characters")
                return content
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Media download and parsing failed: {e}")
            return f"Error processing media: {str(e)}"

# Global handler instance
whatsapp_handler = WhatsAppHandler()

@router.post("/webhook")
async def whatsapp_webhook(
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(""),
    MediaUrl0: Optional[str] = Form(None),
    MessageSid: Optional[str] = Form(None)
):
    """
    Twilio WhatsApp webhook endpoint
    Receives incoming WhatsApp messages and processes them
    """
    try:
        # Prepare message data
        message_data = {
            "From": From,
            "To": To,
            "Body": Body,
            "MediaUrl0": MediaUrl0,
            "MessageSid": MessageSid
        }
        
        logger.info(f"Received WhatsApp message from {From}: {Body[:100]}...")
        
        # Process message in background to avoid timeout
        response_text = await whatsapp_handler.process_message(message_data)
        
        # Create TwiML response
        twiml = MessagingResponse()
        twiml.message(response_text)
        
        # Return response with proper Content-Type header
        return Response(content=str(twiml), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        
        # Return error response with proper Content-Type
        error_twiml = MessagingResponse()
        error_twiml.message("Sorry, I encountered an error processing your message. Please try again later.")
        return Response(content=str(error_twiml), media_type="application/xml")

@router.get("/webhook")
async def webhook_verification():
    """Handle Twilio webhook verification"""
    return {"status": "WhatsApp webhook active", "timestamp": datetime.utcnow().isoformat()}

@router.post("/send-message")
async def send_message(
    to: str,
    message: str
):
    """Send a WhatsApp message (for testing)"""
    try:
        # Send message via handler
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        
        message_instance = client.messages.create(
            body=message,
            from_=config.TWILIO_PHONE_NUMBER,
            to=f"whatsapp:{to}" if not to.startswith("whatsapp:") else to
        )
        
        return {"status": "sent", "message_sid": message_instance.sid}
        
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail=str(e))
