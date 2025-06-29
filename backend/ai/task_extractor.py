"""
Task and Reminder Extractor using IBM Granite-3-3-8b
Intelligently extracts tasks, reminders, and deadlines from natural language text

Example Usage:
INPUT: 
"I need to call John tomorrow at 3pm and remind me to submit the quarterly report by Friday. 
Also, don't forget to review the contract documents next week and schedule a team meeting."

EXPECTED OUTPUT:
{
    "tasks": [
        {
            "title": "Call John",
            "description": "Make phone call to John",
            "due_date": "2025-06-30T15:00:00Z",
            "priority": "medium",
            "category": "communication"
        },
        {
            "title": "Submit quarterly report", 
            "description": "Submit the quarterly report",
            "due_date": "2025-07-04T23:59:00Z",
            "priority": "high",
            "category": "work"
        }
    ],
    "reminders": [
        {
            "title": "Review contract documents",
            "message": "Don't forget to review the contract documents",
            "scheduled_time": "2025-07-07T09:00:00Z",
            "category": "work"
        }
    ]
}
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from backend.ai.granite_api import granite_client
from backend.utils.time_parser import parse_time_expression
from backend.models.schema import TaskItem, ReminderItem

logger = logging.getLogger(__name__)

class TaskExtractor:
    """Advanced task and reminder extraction using Granite-3-3-8b"""
    
    def __init__(self):
        self.granite = granite_client
    
    async def extract_tasks_and_reminders(
        self, 
        text: str, 
        user_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract tasks and reminders from natural language text
        
        Args:
            text: Input text to analyze
            user_id: User identifier for task assignment
            
        Returns:
            Dictionary with extracted tasks and reminders
        """
        try:
            # First pass: Identify potential tasks and reminders
            raw_extraction = await self._extract_raw_items(text)
            
            # Second pass: Parse dates and times
            processed_tasks = await self._process_tasks(raw_extraction.get("tasks", []), user_id)
            processed_reminders = await self._process_reminders(raw_extraction.get("reminders", []), user_id)
            
            return {
                "tasks": processed_tasks,
                "reminders": processed_reminders
            }
            
        except Exception as e:
            logger.error(f"Task extraction failed: {e}")
            return {"tasks": [], "reminders": []}
    
    async def _extract_raw_items(self, text: str) -> Dict[str, List[Dict]]:
        """Extract raw tasks and reminders using Granite"""
        
        prompt = f"""
Analyze the following text and extract any tasks, todos, reminders, or action items mentioned.
Return the results in this exact JSON format:

{{
    "tasks": [
        {{
            "title": "brief task title",
            "description": "detailed task description", 
            "due_expression": "natural language time expression (e.g., 'tomorrow at 3pm', 'Friday', 'next week')",
            "priority": "high|medium|low",
            "category": "work|personal|health|finance|communication|other"
        }}
    ],
    "reminders": [
        {{
            "title": "reminder title",
            "message": "full reminder message",
            "time_expression": "when to remind (e.g., 'tomorrow morning', 'in 2 hours', 'next Monday')",
            "category": "work|personal|health|finance|communication|other"
        }}
    ]
}}

Look for keywords like: need to, should, must, remind me, don't forget, deadline, due, schedule, call, email, submit, review, complete, finish, etc.

Text: {text}

JSON:"""
        
        try:
            response = await self.granite.generate_text(prompt, max_tokens=500, temperature=0.3)
            
            # Extract JSON from response
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
                
        except Exception as e:
            logger.warning(f"Raw extraction parsing failed: {e}")
        
        # Fallback: Try regex-based extraction
        return await self._fallback_extraction(text)
    
    async def _fallback_extraction(self, text: str) -> Dict[str, List[Dict]]:
        """Fallback extraction using pattern matching"""
        import re
        
        tasks = []
        reminders = []
        
        # Task patterns
        task_patterns = [
            r"(?:need to|should|must|have to)\s+([^.!?]+)",
            r"(?:todo|to do):\s*([^.!?]+)",
            r"(?:task|action item):\s*([^.!?]+)"
        ]
        
        # Reminder patterns  
        reminder_patterns = [
            r"remind me (?:to\s+)?([^.!?]+)",
            r"don't forget (?:to\s+)?([^.!?]+)",
            r"remember to\s+([^.!?]+)"
        ]
        
        # Extract tasks
        for pattern in task_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                task_text = match.group(1).strip()
                if len(task_text) > 3:
                    tasks.append({
                        "title": task_text[:50],
                        "description": task_text,
                        "due_expression": "no specific time",
                        "priority": "medium",
                        "category": "other"
                    })
        
        # Extract reminders
        for pattern in reminder_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                reminder_text = match.group(1).strip()
                if len(reminder_text) > 3:
                    reminders.append({
                        "title": f"Reminder: {reminder_text[:30]}",
                        "message": f"Don't forget to {reminder_text}",
                        "time_expression": "no specific time",
                        "category": "other"
                    })
        
        return {"tasks": tasks[:5], "reminders": reminders[:5]}  # Limit results
    
    async def _process_tasks(self, raw_tasks: List[Dict], user_id: str) -> List[Dict[str, Any]]:
        """Process raw tasks into structured format"""
        processed = []
        
        for raw_task in raw_tasks:
            try:
                # Parse due date
                due_date = None
                if raw_task.get("due_expression") and raw_task["due_expression"] != "no specific time":
                    due_date = parse_time_expression(raw_task["due_expression"])
                
                task = {
                    "task_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "title": raw_task.get("title", "")[:100],
                    "description": raw_task.get("description", ""),
                    "due_date": due_date.isoformat() if due_date else None,
                    "priority": raw_task.get("priority", "medium"),
                    "category": raw_task.get("category", "other"),
                    "completed": False,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                processed.append(task)
                
            except Exception as e:
                logger.warning(f"Failed to process task: {raw_task}, error: {e}")
                continue
        
        return processed
    
    async def _process_reminders(self, raw_reminders: List[Dict], user_id: str) -> List[Dict[str, Any]]:
        """Process raw reminders into structured format"""
        processed = []
        
        for raw_reminder in raw_reminders:
            try:
                # Parse reminder time
                scheduled_time = None
                if raw_reminder.get("time_expression") and raw_reminder["time_expression"] != "no specific time":
                    scheduled_time = parse_time_expression(raw_reminder["time_expression"])
                else:
                    # Default to 1 hour from now
                    scheduled_time = datetime.now(timezone.utc) + timedelta(hours=1)
                
                reminder = {
                    "reminder_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "title": raw_reminder.get("title", "")[:100],
                    "message": raw_reminder.get("message", ""),
                    "scheduled_time": scheduled_time.isoformat() if scheduled_time else None,
                    "category": raw_reminder.get("category", "other"),
                    "status": "pending",
                    "created_at": datetime.utcnow().isoformat()
                }
                
                processed.append(reminder)
                
            except Exception as e:
                logger.warning(f"Failed to process reminder: {raw_reminder}, error: {e}")
                continue
        
        return processed
    
    async def extract_deadlines(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract specific deadlines and due dates
        
        Example:
        INPUT: "Report due Friday, presentation next Tuesday, meeting deadline end of month"
        OUTPUT: [
            {"item": "Report", "deadline": "2025-07-04T23:59:00Z", "urgency": "high"},
            {"item": "Presentation", "deadline": "2025-07-08T23:59:00Z", "urgency": "medium"}
        ]
        """
        prompt = f"""
Extract any deadlines, due dates, or time-sensitive items from this text.
Return JSON format:
[
    {{
        "item": "what is due",
        "deadline_expression": "natural language time",
        "urgency": "high|medium|low"
    }}
]

Text: {text}

JSON:"""
        
        try:
            response = await self.granite.generate_text(prompt, max_tokens=300, temperature=0.3)
            
            import json
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                raw_deadlines = json.loads(json_str)
                
                # Process deadlines
                processed = []
                for deadline in raw_deadlines:
                    parsed_time = parse_time_expression(deadline.get("deadline_expression", ""))
                    if parsed_time:
                        processed.append({
                            "item": deadline.get("item", ""),
                            "deadline": parsed_time.isoformat(),
                            "urgency": deadline.get("urgency", "medium")
                        })
                
                return processed
                
        except Exception as e:
            logger.warning(f"Deadline extraction failed: {e}")
        
        return []
    
    async def prioritize_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """
        Analyze and prioritize tasks based on urgency and importance
        
        Example:
        INPUT: [{"title": "Submit tax report", "due_date": "2025-06-30"}, {"title": "Read book", "due_date": null}]
        OUTPUT: [{"title": "Submit tax report", "priority": "high", "score": 9}, {"title": "Read book", "priority": "low", "score": 3}]
        """
        if not tasks:
            return []
        
        task_list = "\n".join([f"- {task.get('title', '')}: due {task.get('due_date', 'no deadline')}" for task in tasks])
        
        prompt = f"""
Analyze these tasks and assign priority scores (1-10) based on urgency and importance.
Return JSON format:
[
    {{
        "title": "task title",
        "priority": "high|medium|low", 
        "score": 8,
        "reasoning": "why this priority"
    }}
]

Tasks:
{task_list}

JSON:"""
        
        try:
            response = await self.granite.generate_text(prompt, max_tokens=400, temperature=0.3)
            
            import json
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                priorities = json.loads(json_str)
                
                # Merge with original tasks
                for i, task in enumerate(tasks):
                    if i < len(priorities):
                        task.update(priorities[i])
                
                # Sort by score
                return sorted(tasks, key=lambda x: x.get('score', 5), reverse=True)
                
        except Exception as e:
            logger.warning(f"Task prioritization failed: {e}")
        
        return tasks

# Global extractor instance
task_extractor = TaskExtractor()

# Test function
async def test_task_extractor():
    """Test task extraction functionality"""
    
    test_texts = [
        "I need to call John tomorrow at 3pm and remind me to submit the quarterly report by Friday.",
        "Don't forget to review the contract documents next week and schedule a team meeting.",
        "Todo: finish the presentation, send emails to clients, and pay bills by end of month.",
        "Deadline: project proposal due Monday, team review meeting Tuesday at 2pm."
    ]
    
    print("🧪 Testing Task Extractor")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        try:
            result = await task_extractor.extract_tasks_and_reminders(text, "test_user")
            print(f"✅ Tasks: {len(result['tasks'])}")
            for task in result['tasks']:
                print(f"   - {task['title']} (due: {task.get('due_date', 'TBD')})")
            
            print(f"✅ Reminders: {len(result['reminders'])}")
            for reminder in result['reminders']:
                print(f"   - {reminder['title']} (at: {reminder.get('scheduled_time', 'TBD')})")
                
        except Exception as e:
            print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_task_extractor())
