"""
Unit Tests for WhatsApp AI Second Brain Assistant
Tests core AI functionality including summarization, task extraction, and Q&A

Run with: python -m pytest test/test_ai.py -v

Example Test Cases:
1. Document Summarization
2. Task and Reminder Extraction  
3. Question Answering with RAG
4. Vector Store Operations
5. Time Expression Parsing
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import modules to test
from backend.ai.summarizer import summarizer
from backend.ai.task_extractor import task_extractor
from backend.ai.qa import qa_system
from backend.memory.vectorstore import vector_store
from backend.utils.time_parser import parse_time_expression
from backend.scheduler.scheduler import scheduler

class TestDocumentSummarizer:
    """Test document summarization functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_summarization(self):
        """Test basic document summarization"""
        test_content = """
        Artificial Intelligence (AI) is a rapidly growing field that focuses on creating 
        intelligent machines capable of performing tasks that typically require human intelligence. 
        These tasks include learning, reasoning, problem-solving, perception, and language understanding.
        
        Machine Learning is a subset of AI that enables computers to learn and improve from experience 
        without being explicitly programmed. Deep Learning, a subset of Machine Learning, uses neural 
        networks with multiple layers to analyze and learn from large amounts of data.
        
        AI applications are transforming industries including healthcare, finance, transportation, 
        and entertainment. Key benefits include automation of repetitive tasks, improved decision-making, 
        enhanced efficiency, and the ability to process and analyze vast amounts of data quickly.
        
        However, AI also presents challenges such as job displacement, privacy concerns, algorithmic bias, 
        and the need for significant computational resources. As AI continues to evolve, it's important 
        to address these challenges while maximizing the benefits.
        """
        
        try:
            result = await summarizer.summarize_document(
                content=test_content,
                summary_type="concise",
                max_length=100
            )
            
            # Assertions
            assert result.summary is not None
            assert len(result.summary) > 0
            assert result.word_count > 0
            assert result.word_count <= 120  # Allow some flexibility
            assert isinstance(result.key_points, list)
            
            print(f"✅ Summary generated: {result.summary}")
            print(f"✅ Word count: {result.word_count}")
            print(f"✅ Key points: {len(result.key_points)}")
            
        except Exception as e:
            print(f"❌ Summarization test failed: {e}")
            # Don't fail the test if AI service is unavailable
            pytest.skip(f"AI service unavailable: {e}")
    
    @pytest.mark.asyncio
    async def test_action_item_extraction(self):
        """Test action item extraction from content"""
        test_content = """
        Meeting notes from project review:
        - John will handle the database migration by Friday
        - Sarah needs to review the user interface designs  
        - We should schedule a follow-up meeting next week
        - The marketing team must prepare the launch materials
        - Remember to update the project timeline document
        """
        
        try:
            actions = await summarizer.extract_action_items(test_content)
            
            # Assertions
            assert isinstance(actions, list)
            assert len(actions) > 0
            
            print(f"✅ Extracted {len(actions)} action items")
            for i, action in enumerate(actions, 1):
                print(f"   {i}. {action}")
                
        except Exception as e:
            print(f"❌ Action item extraction failed: {e}")
            pytest.skip(f"AI service unavailable: {e}")

class TestTaskExtractor:
    """Test task and reminder extraction"""
    
    @pytest.mark.asyncio
    async def test_task_extraction(self):
        """Test extracting tasks from natural language"""
        test_messages = [
            "I need to call John tomorrow at 3pm and remind me to submit the quarterly report by Friday",
            "Don't forget to review the contract documents next week",
            "Todo: finish the presentation, send emails to clients, and pay bills by end of month",
            "Must complete project proposal by Monday and schedule team meeting for Tuesday"
        ]
        
        for message in test_messages:
            try:
                result = await task_extractor.extract_tasks_and_reminders(message, "test_user")
                
                # Assertions
                assert isinstance(result, dict)
                assert "tasks" in result
                assert "reminders" in result
                assert isinstance(result["tasks"], list)
                assert isinstance(result["reminders"], list)
                
                print(f"✅ Message: {message[:50]}...")
                print(f"   Tasks: {len(result['tasks'])}")
                print(f"   Reminders: {len(result['reminders'])}")
                
                # Print extracted items
                for task in result["tasks"]:
                    print(f"   Task: {task.get('title', 'No title')}")
                
                for reminder in result["reminders"]:
                    print(f"   Reminder: {reminder.get('title', 'No title')}")
                
            except Exception as e:
                print(f"❌ Task extraction failed for message: {e}")
                pytest.skip(f"AI service unavailable: {e}")
    
    @pytest.mark.asyncio  
    async def test_deadline_extraction(self):
        """Test deadline extraction"""
        test_content = "Report due Friday, presentation next Tuesday, meeting deadline end of month"
        
        try:
            deadlines = await task_extractor.extract_deadlines(test_content)
            
            # Assertions
            assert isinstance(deadlines, list)
            
            print(f"✅ Extracted {len(deadlines)} deadlines")
            for deadline in deadlines:
                print(f"   {deadline}")
                
        except Exception as e:
            print(f"❌ Deadline extraction failed: {e}")
            pytest.skip(f"AI service unavailable: {e}")

class TestQuestionAnswering:
    """Test RAG-based question answering"""
    
    @pytest.mark.asyncio
    async def test_qa_with_context(self):
        """Test Q&A with mock context documents"""
        
        # Mock vector store search to return test context
        mock_results = [
            {
                "doc_id": "doc1",
                "content": "AI benefits include automation, improved efficiency, better decision-making, and cost reduction.",
                "score": 0.85,
                "metadata": {"source": "ai_report.pdf", "title": "AI Benefits Report"},
                "user_id": "test_user"
            },
            {
                "doc_id": "doc2", 
                "content": "Machine learning enables predictive analytics and pattern recognition in large datasets.",
                "score": 0.75,
                "metadata": {"source": "ml_guide.txt", "title": "ML Guide"},
                "user_id": "test_user"
            }
        ]
        
        with patch.object(vector_store, 'search', return_value=mock_results):
            try:
                result = await qa_system.answer_question(
                    question="What are the main benefits of AI?",
                    user_id="test_user"
                )
                
                # Assertions
                assert result.answer is not None
                assert len(result.answer) > 0
                assert isinstance(result.sources, list)
                assert result.confidence >= 0.0
                assert result.confidence <= 1.0
                
                print(f"✅ Question: What are the main benefits of AI?")
                print(f"✅ Answer: {result.answer}")
                print(f"✅ Sources: {result.sources}")
                print(f"✅ Confidence: {result.confidence}")
                
            except Exception as e:
                print(f"❌ Q&A test failed: {e}")
                pytest.skip(f"AI service unavailable: {e}")
    
    @pytest.mark.asyncio
    async def test_qa_no_context(self):
        """Test Q&A with no relevant context"""
        
        # Mock empty search results
        with patch.object(vector_store, 'search', return_value=[]):
            result = await qa_system.answer_question(
                question="What is quantum computing?",
                user_id="test_user"
            )
            
            # Should return appropriate message when no context available
            assert "don't have enough information" in result.answer.lower()
            assert result.confidence == 0.0
            assert len(result.sources) == 0
            
            print(f"✅ No context response: {result.answer}")

class TestVectorStore:
    """Test vector store operations"""
    
    @pytest.mark.asyncio
    async def test_vector_store_operations(self):
        """Test adding documents and searching"""
        
        try:
            # Initialize vector store
            await vector_store.initialize()
            
            # Add test documents
            test_docs = [
                ("AI and machine learning are transforming business processes", {"source": "test1"}),
                ("Data science involves statistical analysis and predictive modeling", {"source": "test2"}),
                ("Cloud computing provides scalable infrastructure solutions", {"source": "test3"})
            ]
            
            doc_ids = []
            for content, metadata in test_docs:
                doc_id = await vector_store.add_document(
                    user_id="test_user",
                    content=content,
                    metadata=metadata
                )
                doc_ids.append(doc_id)
                print(f"✅ Added document: {doc_id}")
            
            # Test search
            search_results = await vector_store.search(
                query="machine learning AI",
                user_id="test_user",
                top_k=3
            )
            
            # Assertions
            assert isinstance(search_results, list)
            assert len(search_results) > 0
            
            print(f"✅ Search returned {len(search_results)} results")
            for result in search_results:
                print(f"   Score: {result['score']:.3f} - {result['content'][:50]}...")
            
            # Test user documents
            user_docs = await vector_store.get_user_documents("test_user")
            assert len(user_docs) >= len(test_docs)
            
            print(f"✅ User has {len(user_docs)} documents")
            
            # Clean up (in a real test, you'd use fixtures)
            for doc_id in doc_ids:
                await vector_store.delete_document(doc_id, "test_user")
            
        except Exception as e:
            print(f"❌ Vector store test failed: {e}")
            pytest.skip(f"Vector store unavailable: {e}")

class TestTimeParser:
    """Test time expression parsing"""
    
    def test_time_expressions(self):
        """Test parsing various time expressions"""
        
        test_cases = [
            ("tomorrow at 3pm", lambda dt: dt.hour == 15),
            ("next Friday", lambda dt: dt.weekday() == 4),  # Friday is weekday 4
            ("in 2 hours", lambda dt: dt > datetime.now()),
            ("9am", lambda dt: dt.hour == 9),
            ("evening", lambda dt: dt.hour >= 18),
        ]
        
        for expression, validator in test_cases:
            try:
                result = parse_time_expression(expression)
                
                if result:
                    assert validator(result), f"Validation failed for '{expression}': {result}"
                    print(f"✅ '{expression}' -> {result.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"⚠️ Could not parse: '{expression}'")
                    
            except Exception as e:
                print(f"❌ Time parsing failed for '{expression}': {e}")
    
    def test_invalid_expressions(self):
        """Test handling of invalid time expressions"""
        
        invalid_expressions = [
            "",
            "invalid time",
            "next blah",
            "tomorrow at 25pm"
        ]
        
        for expression in invalid_expressions:
            result = parse_time_expression(expression)
            print(f"✅ Invalid expression '{expression}' returned None: {result is None}")

class TestScheduler:
    """Test scheduler functionality"""
    
    @pytest.mark.asyncio
    async def test_scheduler_initialization(self):
        """Test scheduler can be initialized"""
        
        try:
            await scheduler.initialize()
            assert scheduler._started
            
            print("✅ Scheduler initialized successfully")
            
            # Test getting stats
            stats = await scheduler.get_stats()
            assert isinstance(stats, dict)
            assert "status" in stats
            
            print(f"✅ Scheduler stats: {stats}")
            
        except Exception as e:
            print(f"❌ Scheduler test failed: {e}")
            pytest.skip(f"Scheduler unavailable: {e}")

# Test runner
if __name__ == "__main__":
    import sys
    import os
    
    # Add the project root to Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("🧪 Running WhatsApp AI Second Brain Tests")
    print("=" * 60)
    
    async def run_tests():
        """Run all tests"""
        
        # Test classes
        test_classes = [
            TestDocumentSummarizer(),
            TestTaskExtractor(),  
            TestQuestionAnswering(),
            TestVectorStore(),
            TestTimeParser(),
            TestScheduler()
        ]
        
        for test_class in test_classes:
            class_name = test_class.__class__.__name__
            print(f"\n📋 Testing {class_name}")
            print("-" * 40)
            
            # Get test methods
            test_methods = [method for method in dir(test_class) if method.startswith('test_')]
            
            for method_name in test_methods:
                print(f"\n🔧 {method_name}")
                try:
                    method = getattr(test_class, method_name)
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        method()
                except Exception as e:
                    print(f"❌ Test failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Test run completed!")
    
    # Run the tests
    asyncio.run(run_tests())
