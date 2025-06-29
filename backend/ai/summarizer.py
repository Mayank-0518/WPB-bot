"""
Document Summarizer using IBM Granite-3-3-8b
Provides intelligent summarization of text content, documents, and messages

Example Usage:
INPUT: 
"""
"This quarterly report shows significant growth in our AI division. "
"Revenue increased by 45% compared to last quarter, primarily driven by "
"enterprise AI solutions. The team expanded from 12 to 18 engineers."
"Key challenges include talent acquisition and scaling infrastructure."
"""

EXPECTED OUTPUT:
{
    "summary": "Q2 AI division report: 45% revenue growth from enterprise solutions, team grew 50%, facing hiring and scaling challenges.",
    "key_points": [
        "AI division revenue up 45%",
        "Enterprise solutions driving growth", 
        "Team expanded from 12 to 18 engineers",
        "Challenges: talent acquisition, infrastructure scaling"
    ],
    "word_count": 28
}
"""

import logging
from typing import Dict, List, Any
from backend.ai.granite_api import granite_client
from backend.models.schema import SummaryRequest, SummaryResponse

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """Advanced document summarization using Granite-3-3-8b"""
    
    def __init__(self):
        self.granite = granite_client
    
    async def summarize_document(
        self, 
        content: str, 
        summary_type: str = "concise",
        max_length: int = 200
    ) -> SummaryResponse:
        """
        Generate comprehensive document summary
        
        Args:
            content: Text content to summarize
            summary_type: Type of summary (concise, detailed, bullet_points)
            max_length: Maximum words in summary
            
        Returns:
            SummaryResponse with summary, key points, and metadata
        """
        try:
            # Generate main summary
            summary = await self._generate_summary(content, summary_type, max_length)
            
            # Extract key points
            key_points = await self._extract_key_points(content)
            
            # Count words
            word_count = len(summary.split())
            
            return SummaryResponse(
                summary=summary,
                key_points=key_points,
                word_count=word_count
            )
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return SummaryResponse(
                summary="Unable to generate summary at this time.",
                key_points=[],
                word_count=0
            )
    
    async def _generate_summary(
        self, 
        content: str, 
        summary_type: str, 
        max_length: int
    ) -> str:
        """Generate summary based on type"""
        
        prompts = {
            "concise": f"""
Provide a concise summary of the following text in {max_length} words or less. 
Focus on the most important information and main takeaways.

Text: {content[:4000]}

Concise Summary:""",
            
            "detailed": f"""
Provide a detailed summary of the following text in {max_length} words or less.
Include important details, context, and implications.

Text: {content[:4000]}

Detailed Summary:""",
            
            "bullet_points": f"""
Summarize the following text as bullet points, maximum {max_length} words total.
Each bullet should capture a key point or insight.

Text: {content[:4000]}

Key Points:
•""",
            
            "executive": f"""
Provide an executive summary of the following text in {max_length} words or less.
Focus on key decisions, outcomes, and strategic implications.

Text: {content[:4000]}

Executive Summary:"""
        }
        
        prompt = prompts.get(summary_type, prompts["concise"])
        return await self.granite.generate_text(prompt, max_tokens=max_length*2)
    
    async def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        prompt = f"""
Extract the 3-5 most important key points from this text.
Return each point as a concise statement.

Text: {content[:3000]}

Key Points:
1."""
        
        try:
            response = await self.granite.generate_text(prompt, max_tokens=200)
            
            # Parse numbered points
            points = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Look for numbered points (1., 2., etc.) or bullet points
                if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '•', '-']):
                    # Clean up the point
                    point = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '•', '-']:
                        point = point.replace(prefix, '').strip()
                    if point and len(point) > 10:  # Filter out very short points
                        points.append(point)
            
            return points[:5]  # Return max 5 points
            
        except Exception as e:
            logger.warning(f"Key point extraction failed: {e}")
            return []
    
    async def summarize_conversation(self, messages: List[str]) -> str:
        """
        Summarize a conversation or chat history
        
        Example:
        INPUT: ["Hi, can you help with the report?", "Sure, what do you need?", "I need a summary by tomorrow"]
        OUTPUT: "User requested help with a report summary due tomorrow."
        """
        conversation_text = "\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(messages)])
        
        prompt = f"""
Summarize this conversation in 2-3 sentences. Focus on the main topic and any action items.

Conversation:
{conversation_text[:2000]}

Summary:"""
        
        return await self.granite.generate_text(prompt, max_tokens=100)
    
    async def extract_action_items(self, content: str) -> List[str]:
        """
        Extract action items from text
        
        Example:
        INPUT: "We discussed the project timeline. John will handle design by Friday. Sarah needs to review contracts."
        OUTPUT: ["John will handle design by Friday", "Sarah needs to review contracts"]
        """
        prompt = f"""
Extract any action items, tasks, or next steps mentioned in this text.
Return each action item on a separate line starting with "- ".

Text: {content[:3000]}

Action Items:
-"""
        
        try:
            response = await self.granite.generate_text(prompt, max_tokens=200)
            
            # Parse action items
            items = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    item = line[1:].strip()
                    if item and len(item) > 5:
                        items.append(item)
            
            return items
            
        except Exception:
            return []
    
    async def categorize_content(self, content: str) -> Dict[str, Any]:
        """
        Categorize content by type and topic
        
        Example:
        INPUT: "Meeting notes from quarterly review discussing revenue, expenses, and team performance."
        OUTPUT: {
            "category": "meeting_notes",
            "topics": ["finance", "performance", "review"],
            "document_type": "business"
        }
        """
        prompt = f"""
Analyze and categorize this content. Return JSON format:
{{
    "category": "meeting_notes|email|report|document|note|other",
    "topics": ["topic1", "topic2", "topic3"],
    "document_type": "business|personal|technical|academic|other",
    "urgency": "high|medium|low"
}}

Content: {content[:2000]}

JSON:"""
        
        try:
            response = await self.granite.generate_text(prompt, max_tokens=150, temperature=0.3)
            
            # Try to parse JSON
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
                
        except Exception as e:
            logger.warning(f"Content categorization failed: {e}")
        
        # Fallback
        return {
            "category": "document",
            "topics": ["general"],
            "document_type": "other",
            "urgency": "medium"
        }

# Global summarizer instance
summarizer = DocumentSummarizer()

# Example usage and testing
async def test_summarizer():
    """Test summarizer functionality"""
    
    test_content = """
    This quarterly report shows significant growth in our AI division. 
    Revenue increased by 45% compared to last quarter, primarily driven by 
    enterprise AI solutions. The team expanded from 12 to 18 engineers.
    Key challenges include talent acquisition and scaling infrastructure.
    We need to focus on hiring senior developers and investing in cloud infrastructure.
    The next quarter goals are to maintain growth momentum and improve operational efficiency.
    """
    
    print("🧪 Testing Document Summarizer")
    print("=" * 50)
    
    # Test summary generation
    try:
        result = await summarizer.summarize_document(test_content, "concise", 50)
        print(f"✅ Summary: {result.summary}")
        print(f"✅ Key Points: {result.key_points}")
        print(f"✅ Word Count: {result.word_count}")
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
    
    # Test action item extraction
    try:
        actions = await summarizer.extract_action_items(test_content)
        print(f"✅ Action Items: {actions}")
    except Exception as e:
        print(f"❌ Action extraction failed: {e}")
    
    # Test categorization
    try:
        category = await summarizer.categorize_content(test_content)
        print(f"✅ Categorization: {category}")
    except Exception as e:
        print(f"❌ Categorization failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_summarizer())
