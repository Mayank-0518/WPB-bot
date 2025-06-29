"""
Main FastAPI Application for WhatsApp AI Second Brain Assistant
Entry point for the application with all routes and middleware configured

Run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Features:
- WhatsApp webhook handling via Twilio
- Document summarization using IBM Granite-3-3-8b
- RAG-based question answering
- Task and reminder extraction
- Scheduled reminders via APScheduler
- Vector search with FAISS
- File upload and parsing
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from backend.utils.config import config
from backend.routes.whatsapp import router as whatsapp_router
from backend.routes.reminders import router as reminders_router
from backend.ai.granite_api import granite_client
from backend.ai.summarizer import summarizer
from backend.ai.qa import qa_system
from backend.memory.vectorstore import vector_store
from backend.scheduler.scheduler import scheduler
from backend.files.parser import file_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting WhatsApp AI Second Brain Assistant...")
    
    try:
        # Validate configuration
        config.validate()
        logger.info("Configuration validated")
        
        # Initialize services
        await vector_store.initialize()
        logger.info("Vector store initialized")
        
        await scheduler.initialize()
        logger.info("Scheduler initialized")
        
        logger.info("🚀 WhatsApp AI Second Brain Assistant is ready!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        await scheduler.shutdown()
        logger.info("Services shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="WhatsApp AI Second Brain Assistant",
    description="An intelligent assistant that helps you manage information, set reminders, and answer questions via WhatsApp",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(whatsapp_router, prefix="/whatsapp")
app.include_router(reminders_router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "WhatsApp AI Second Brain Assistant",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "whatsapp_webhook": "/whatsapp/webhook",
            "reminders": "/api/reminders",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "WhatsApp integration via Twilio",
            "Document summarization with IBM Granite-3-3-8b",
            "RAG-based question answering",
            "Task and reminder extraction",
            "Scheduled notifications",
            "Vector search with FAISS",
            "File upload and parsing"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check core services
        services_status = {
            "vector_store": vector_store._initialized,
            "scheduler": scheduler._started,
            "granite_api": config.GRANITE_API_KEY is not None,
            "twilio": all([config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN, config.TWILIO_PHONE_NUMBER])
        }
        
        # Get system stats
        stats = {
            "vector_store": await vector_store.get_stats() if vector_store._initialized else {},
            "scheduler": await scheduler.get_stats() if scheduler._started else {}
        }
        
        all_healthy = all(services_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """Upload and process a file"""
    try:
        # Validate file
        if not file_parser.is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )
        
        # Read file content
        content = await file.read()
        
        # Save file
        file_path = await file_parser.save_uploaded_file(content, file.filename)
        
        # Parse file
        parsed_result = await file_parser.parse_file(file_path)
        
        # Store in vector database
        doc_id = await vector_store.add_document(
            user_id=user_id,
            content=parsed_result["content"],
            metadata={
                "source": "file_upload",
                "filename": file.filename,
                "file_type": parsed_result["file_type"],
                "file_size": parsed_result["file_size"],
                "word_count": parsed_result["word_count"],
                "upload_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "file_type": parsed_result["file_type"],
            "content_preview": parsed_result["content"][:200] + "..." if len(parsed_result["content"]) > 200 else parsed_result["content"],
            "word_count": parsed_result["word_count"],
            "message": "File uploaded and processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize_text(request: dict):
    """Summarize text content"""
    try:
        text = request.get("text", "")
        summary_type = request.get("summary_type", "concise")
        max_length = request.get("max_length", 200)
        
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Text is too short to summarize (minimum 50 characters)"
            )
        
        summary_result = await summarizer.summarize_document(
            content=text,
            summary_type=summary_type,
            max_length=max_length
        )
        
        return {
            "success": True,
            "summary": summary_result.summary,
            "key_points": summary_result.key_points,
            "word_count": summary_result.word_count,
            "original_length": len(text.split()),
            "compression_ratio": round(summary_result.word_count / len(text.split()), 2) if len(text.split()) > 0 else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
@app.post("/api/ask")
async def ask_question(request: dict):
    """Ask a question and get an answer based on stored documents"""
    try:
        question = request.get("question", "")
        user_id = request.get("user_id", "default_user")
        context_limit = request.get("context_limit", 5)
        
        if not question or len(question.strip()) < 5:
            raise HTTPException(
                status_code=400,
                detail="Question is too short (minimum 5 characters)"
            )
        
        qa_result = await qa_system.answer_question(
            question=question,
            user_id=user_id,
            context_limit=context_limit
        )
        
        return {
            "success": True,
            "question": question,
            "answer": qa_result.answer,
            "sources": qa_result.sources,
            "confidence": qa_result.confidence,
            "context_documents": len(qa_result.sources)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_documents(
    query: str,
    user_id: str = "default_user",
    limit: int = 10
):
    """Search documents using vector similarity"""
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query is too short (minimum 3 characters)"
            )
        
        search_results = await vector_store.search(
            query=query,
            user_id=user_id,
            top_k=limit
        )
        
        return {
            "success": True,
            "query": query,
            "results_count": len(search_results),
            "results": [
                {
                    "doc_id": result["doc_id"],
                    "content_preview": result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"],
                    "score": result["score"],
                    "metadata": result["metadata"]
                }
                for result in search_results
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "vector_store": await vector_store.get_stats() if vector_store._initialized else {},
            "scheduler": await scheduler.get_stats() if scheduler._started else {},
            "config": {
                "debug": config.DEBUG,
                "host": config.HOST,
                "port": config.PORT,
                "timezone": config.SCHEDULER_TIMEZONE
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-ai")
async def test_ai_services():
    """Test AI services connectivity"""
    try:
        # Test Granite API
        test_text = "This is a test message for the AI services."
        
        granite_response = await granite_client.generate_text(
            "Respond with 'AI services are working correctly' if you can process this message.",
            max_tokens=50
        )
        
        # Test summarization
        summary_result = await summarizer.summarize_document(
            "This is a test document about artificial intelligence and machine learning technologies. " * 10,
            "concise",
            50
        )
        
        return {
            "success": True,
            "granite_api": {
                "status": "working",
                "response": granite_response[:100] + "..." if len(granite_response) > 100 else granite_response
            },
            "summarizer": {
                "status": "working",
                "summary_length": summary_result.word_count,
                "key_points_count": len(summary_result.key_points)
            },
            "vector_store": {
                "status": "working" if vector_store._initialized else "not initialized",
                "document_count": await vector_store.get_document_count() if vector_store._initialized else 0
            }
        }
        
    except Exception as e:
        logger.error(f"AI services test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": str(e),
                "message": "One or more AI services are not working properly"
            }
        )

# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.DEBUG else "An error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting WhatsApp AI Second Brain Assistant...")
    
    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )
