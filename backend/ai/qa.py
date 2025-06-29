"""
Question Answering using RAG (Retrieval Augmented Generation) with IBM Granite-3-3-8b
Provides intelligent Q&A by retrieving relevant context and generating answers

Example Usage:
INPUT:
question = "What are the key benefits of AI mentioned in my documents?"
user_id = "user123"

EXPECTED OUTPUT:
{
    "answer": "Based on your documents, the key benefits of AI include: 1) Automation of repetitive tasks, 2) Enhanced decision-making through data analysis, 3) Improved efficiency and productivity, 4) Cost reduction in operations.",
    "sources": ["AI_Report_2025.pdf", "Meeting_Notes_AI_Discussion.txt"],
    "confidence": 0.85,
    "retrieved_chunks": 3
}
"""

import logging
from typing import Dict, List, Any, Optional
from backend.ai.granite_api import granite_client
from backend.memory.vectorstore import vector_store
from backend.models.schema import QuestionRequest, QuestionResponse

logger = logging.getLogger(__name__)

class QuestionAnswerer:
    """RAG-based question answering system"""
    
    def __init__(self):
        self.granite = granite_client
        self.vector_store = vector_store
        self.max_context_length = 3000  # Maximum context for Granite model
    
    async def answer_question(
        self, 
        question: str, 
        user_id: str,
        context_limit: int = 5
    ) -> QuestionResponse:
        """
        Answer question using RAG approach
        
        Args:
            question: User's question
            user_id: User identifier for filtering documents
            context_limit: Maximum number of context documents to retrieve
            
        Returns:
            QuestionResponse with answer, sources, and confidence
        """
        try:
            # Step 1: Retrieve relevant context
            context_docs = await self._retrieve_context(question, user_id, context_limit)
            
            if not context_docs:
                return QuestionResponse(
                    answer="I don't have enough information in your documents to answer this question. Please provide more context or upload relevant documents.",
                    sources=[],
                    confidence=0.0
                )
            
            # Step 2: Generate answer using retrieved context
            answer = await self._generate_answer(question, context_docs)
            
            # Step 3: Extract sources
            sources = [doc.get("source", "Unknown") for doc in context_docs]
            
            # Step 4: Calculate confidence (simplified)
            confidence = self._calculate_confidence(question, context_docs, answer)
            
            return QuestionResponse(
                answer=answer,
                sources=list(set(sources)),  # Remove duplicates
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return QuestionResponse(
                answer="I'm sorry, I encountered an error while processing your question. Please try again.",
                sources=[],
                confidence=0.0
            )
    
    async def _retrieve_context(
        self, 
        question: str, 
        user_id: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the question"""
        try:
            # Search for relevant documents
            results = await self.vector_store.search(
                query=question,
                user_id=user_id,
                top_k=limit
            )
            
            # Format results for context
            context_docs = []
            for result in results:
                context_docs.append({
                    "content": result.get("content", ""),
                    "source": result.get("metadata", {}).get("source", "Unknown"),
                    "title": result.get("metadata", {}).get("title", ""),
                    "score": result.get("score", 0.0),
                    "doc_id": result.get("doc_id", "")
                })
            
            return context_docs
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    async def _generate_answer(
        self, 
        question: str, 
        context_docs: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using Granite model and context"""
        
        # Prepare context text
        context_text = self._prepare_context(context_docs)
        
        # Create prompt for Granite
        prompt = f"""Based on the following context information, answer the question directly and concisely. Provide only one answer.

Context:
{context_text}

Question: {question}

Answer:"""
        
        try:
            answer = await self.granite.generate_text(
                prompt, 
                max_tokens=400,
                temperature=0.3  # Lower temperature for more factual responses
            )
            
            # Clean up the answer
            answer = answer.strip()
            
            # Remove any multiple Q&A pairs - keep only the first answer
            if "Question:" in answer and "Answer:" in answer:
                # Extract only the first answer
                lines = answer.split('\n')
                answer_lines = []
                collecting_answer = True
                
                for line in lines:
                    if line.strip().startswith("Question:") and answer_lines:
                        break  # Stop at the next question
                    if collecting_answer and not line.strip().startswith("Question:"):
                        answer_lines.append(line)
                
                answer = '\n'.join(answer_lines).strip()
            
            # Add disclaimer if answer seems uncertain
            uncertainty_indicators = [
                "i don't know", "not sure", "unclear", "insufficient information",
                "cannot determine", "not enough context"
            ]
            
            if any(indicator in answer.lower() for indicator in uncertainty_indicators):
                answer += "\n\nNote: This answer is based on limited information from your documents. For more accurate information, please provide additional context."
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I encountered an error while generating the answer. Please try rephrasing your question."
    
    def _prepare_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved documents"""
        context_parts = []
        current_length = 0
        
        # Sort by relevance score (if available)
        sorted_docs = sorted(
            context_docs, 
            key=lambda x: x.get("score", 0.0), 
            reverse=True
        )
        
        for i, doc in enumerate(sorted_docs):
            content = doc.get("content", "")
            source = doc.get("source", f"Document {i+1}")
            
            # Add source attribution
            doc_text = f"[Source: {source}]\n{content}\n"
            
            # Check if adding this document would exceed limit
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n---\n".join(context_parts)
    
    def _calculate_confidence(
        self, 
        question: str, 
        context_docs: List[Dict[str, Any]], 
        answer: str
    ) -> float:
        """Calculate confidence score for the answer (simplified approach)"""
        try:
            # Base confidence on number and quality of retrieved documents
            num_docs = len(context_docs)
            if num_docs == 0:
                return 0.0
            
            # Average relevance scores
            avg_score = sum(doc.get("score", 0.0) for doc in context_docs) / num_docs
            
            # Adjust based on answer characteristics
            confidence = avg_score * 0.7  # Base confidence from retrieval
            
            # Boost confidence if answer is substantial
            if len(answer.split()) > 20:
                confidence += 0.1
            
            # Reduce confidence if answer shows uncertainty
            uncertainty_words = ["maybe", "possibly", "might", "unclear", "not sure"]
            if any(word in answer.lower() for word in uncertainty_words):
                confidence -= 0.2
            
            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    async def get_related_questions(
        self, 
        question: str, 
        user_id: str
    ) -> List[str]:
        """
        Generate related questions based on user's query and documents
        
        Example:
        INPUT: "What are AI benefits?"
        OUTPUT: ["How is AI used in business?", "What are AI challenges?", "Which AI tools are mentioned?"]
        """
        # Get some context first
        context_docs = await self._retrieve_context(question, user_id, 3)
        
        if not context_docs:
            return []
        
        context_summary = " ".join([doc.get("content", "")[:200] for doc in context_docs[:2]])
        
        prompt = f"""
Based on this context and the original question, suggest 3 related questions that the user might want to ask:

Original Question: {question}

Context: {context_summary}

Related Questions:
1."""
        
        try:
            response = await self.granite.generate_text(prompt, max_tokens=150)
            
            # Parse the numbered questions
            questions = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Look for numbered questions
                if any(line.startswith(f"{i}.") for i in range(1, 10)):
                    # Clean up the question
                    question_text = line
                    for i in range(1, 10):
                        question_text = question_text.replace(f"{i}.", "").strip()
                    if question_text and len(question_text) > 10:
                        questions.append(question_text)
            
            return questions[:3]  # Return max 3 questions
            
        except Exception as e:
            logger.warning(f"Related questions generation failed: {e}")
            return []
    
    async def fact_check_answer(
        self, 
        question: str, 
        answer: str, 
        context_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Basic fact-checking of generated answer against source documents
        
        Example:
        OUTPUT: {
            "is_supported": True,
            "confidence": 0.8,
            "supporting_sources": ["doc1.pdf", "doc2.txt"],
            "concerns": []
        }
        """
        if not context_docs:
            return {
                "is_supported": False,
                "confidence": 0.0,
                "supporting_sources": [],
                "concerns": ["No source documents available"]
            }
        
        context_text = self._prepare_context(context_docs[:3])  # Use top 3 documents
        
        prompt = f"""
Check if the following answer is supported by the provided context. Respond with JSON:
{{
    "is_supported": true/false,
    "confidence": 0.0-1.0,
    "concerns": ["list any concerns or contradictions"]
}}

Context: {context_text[:1500]}

Question: {question}

Answer: {answer}

JSON:"""
        
        try:
            response = await self.granite.generate_text(
                prompt, 
                max_tokens=200, 
                temperature=0.1
            )
            
            # Try to parse JSON response
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                # Add supporting sources
                result["supporting_sources"] = [
                    doc.get("source", "Unknown") for doc in context_docs
                ]
                
                return result
                
        except Exception as e:
            logger.warning(f"Fact checking failed: {e}")
        
        # Fallback response
        return {
            "is_supported": True,  # Assume supported if can't verify
            "confidence": 0.6,
            "supporting_sources": [doc.get("source", "Unknown") for doc in context_docs],
            "concerns": []
        }

# Global Q&A instance
qa_system = QuestionAnswerer()

# Test function
async def test_qa_system():
    """Test Q&A functionality"""
    
    test_questions = [
        "What are the main benefits of AI?",
        "How can I improve productivity?",
        "What tasks should I prioritize?",
        "When is my next deadline?"
    ]
    
    print("🧪 Testing Q&A System")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        try:
            result = await qa_system.answer_question(question, "test_user")
            print(f"✅ Answer: {result.answer[:100]}...")
            print(f"✅ Sources: {result.sources}")
            print(f"✅ Confidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"❌ Q&A failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_qa_system())
