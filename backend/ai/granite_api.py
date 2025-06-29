"""
IBM Granite-3-3-8b API client for WhatsApp AI Second Brain Assistant
Handles communication with IBM watsonx.ai for text generation and analysis
Uses IAM token authentication for secure access to watsonx.ai services

Example Usage:
INPUT: "Summarize this text: Artificial intelligence is transforming industries..."
OUTPUT: "AI is revolutionizing multiple sectors through automation and intelligent decision-making..."
"""

import asyncio
import httpx
import logging
from typing import Dict, Any, Optional, List
from backend.utils.config import config
from backend.utils.iam_token import get_auth_headers

logger = logging.getLogger(__name__)

class GraniteAPIClient:
    """Client for IBM Granite-3-3-8b model via watsonx.ai with IAM authentication"""
    
    def __init__(self):
        self.api_url = config.GRANITE_API_URL
        self.model_id = config.IBM_MODEL_ID
        self.project_id = config.GRANITE_PROJECT_ID
        
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make async request to Granite API with IAM authentication"""
        try:
            # Get fresh IAM token and headers
            headers = await get_auth_headers()
            
            # Format the correct URL with version parameter
            url = f"{self.api_url}/ml/v1/text/generation?version=2023-05-29"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                
                # Log the response for debugging
                logger.info(f"API Response Status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"API Response Body: {response.text}")
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Authentication failed - check IAM token and project ID")
                # Invalidate token to force refresh on next request
                from backend.utils.iam_token import invalidate_token
                invalidate_token()
            logger.error(f"Granite API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"API request failed: {str(e)}")
        except httpx.HTTPError as e:
            logger.error(f"Granite API request failed: {e}")
            raise Exception(f"API request failed: {str(e)}")
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using Granite model
        
        Example:
        INPUT: "Explain machine learning in simple terms"
        OUTPUT: "Machine learning is like teaching computers to learn patterns..."
        """
        payload = {
            "model_id": self.model_id,
            "project_id": self.project_id,
            "input": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop_sequences": ["</s>", "\n\n---", "User message:", "Response:", "\n\nUser:", "\n\nAssistant:"]
            }
        }
        
        try:
            response = await self._make_request(payload)
            generated_text = response.get("results", [{}])[0].get("generated_text", "")
            
            # Clean up the response
            generated_text = generated_text.strip()
            
            # Remove common artifacts that might leak from prompts
            artifacts_to_remove = [
                "User message:",
                "Response:",
                "Respond naturally and helpfully.",
                "If they need specific functions, guide them on how to use them.",
                "The user says:",
                "User:",
                "Assistant:",
                "AI:",
                "Human:"
            ]
            
            for artifact in artifacts_to_remove:
                if generated_text.startswith(artifact):
                    generated_text = generated_text[len(artifact):].strip()
                # Also remove if it appears after quotes
                if f'"{artifact}' in generated_text:
                    generated_text = generated_text.split(f'"{artifact}')[0].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return "Sorry, I couldn't generate a response right now."
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize long text using Granite
        
        Example:
        INPUT: "Long document about climate change impacts, renewable energy solutions..."
        OUTPUT: "The document discusses climate change effects and renewable energy as mitigation strategies..."
        """
        prompt = f"""
Please provide a concise summary of the following text in {max_length} words or less:

Text: {text[:4000]}  # Limit input to prevent token overflow

Summary:"""
        
        return await self.generate_text(prompt, max_tokens=max_length*2)
    
    async def extract_tasks_and_reminders(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract tasks and reminders from text
        
        Example:
        INPUT: "I need to call John tomorrow and review the report by Friday. Remind me to pay bills next week."
        OUTPUT: {
            "tasks": [{"title": "Call John", "due": "tomorrow"}, {"title": "Review report", "due": "Friday"}],
            "reminders": [{"title": "Pay bills", "when": "next week"}]
        }
        """
        prompt = f"""
Analyze the following text and extract any tasks or reminders mentioned.
Return the results in this exact JSON format:

{{
    "tasks": [
        {{"title": "task description", "due_date": "when it's due", "priority": "high/medium/low"}}
    ],
    "reminders": [
        {{"title": "reminder description", "when": "when to remind", "message": "full reminder text"}}
    ]
}}

Text: {text}

JSON:"""
        
        try:
            response = await self.generate_text(prompt, max_tokens=400, temperature=0.3)
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse task extraction: {e}")
        
        # Fallback response
        return {"tasks": [], "reminders": []}
    
    async def answer_question(self, question: str, context: str) -> str:
        """
        Answer question based on provided context
        
        Example:
        INPUT: 
        question="What are the main benefits of AI?", 
        context="AI document discussing automation, efficiency, decision-making..."
        OUTPUT: "Based on the context, the main benefits of AI include automation of tasks, improved efficiency..."
        """
        prompt = f"""
Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so.

Context: {context[:3000]}

Question: {question}

Answer:"""
        
        return await self.generate_text(prompt, max_tokens=300)
    
    async def categorize_message(self, message: str) -> str:
        """
        Categorize incoming message intent
        
        Example:
        INPUT: "Can you summarize this document for me?"
        OUTPUT: "summarize"
        """
        prompt = f"""Analyze this message and respond with only one word from these options:
- question (user asking something)
- summarize (wants summary/brief)
- reminder (wants to be reminded)
- note (saving information)
- task (mentions todos/tasks)
- general (casual conversation)

Message: "{message}"

Category:"""
        
        response = await self.generate_text(prompt, max_tokens=5, temperature=0.1)
        
        # Clean and validate response
        response = response.lower().strip()
        valid_categories = ["question", "summarize", "reminder", "note", "task", "general"]
        
        # Return the first valid category found, or "general" as fallback
        for category in valid_categories:
            if category in response:
                return category
        
        return "general"

# Global client instance
granite_client = GraniteAPIClient()

# Test function for development
async def test_granite_api():
    """Test the Granite API functionality"""
    test_cases = [
        {
            "function": "generate_text",
            "input": "Explain artificial intelligence in one sentence.",
            "expected": "AI explanation"
        },
        {
            "function": "summarize_text", 
            "input": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
            "expected": "Summary of AI definition"
        },
        {
            "function": "extract_tasks_and_reminders",
            "input": "I need to call the client tomorrow at 3pm and remind me to submit the report by Friday.",
            "expected": "Tasks and reminders extracted"
        }
    ]
    
    for test in test_cases:
        try:
            if test["function"] == "generate_text":
                result = await granite_client.generate_text(test["input"])
            elif test["function"] == "summarize_text":
                result = await granite_client.summarize_text(test["input"])
            elif test["function"] == "extract_tasks_and_reminders":
                result = await granite_client.extract_tasks_and_reminders(test["input"])
            
            print(f"✅ {test['function']}: {result[:100]}...")
        except Exception as e:
            print(f"❌ {test['function']}: {str(e)}")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_granite_api())
