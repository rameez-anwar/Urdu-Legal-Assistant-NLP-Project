import os
import sys

# Fix Urdu text display in terminal
if os.name == 'nt':  # Windows
    os.system("chcp 65001 > nul")
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import logging
from datetime import datetime

import sys
import os

# Add parent directory and src to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.gemini_service import GeminiLegalService
from src.database import LegalDatabase
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal Assistant - Urdu Legal Guidance API",
    description="API for providing legal guidance in Urdu using Google Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
gemini_service = None
db = LegalDatabase()

# Pydantic models
class LegalQuestion(BaseModel):
    question: str
    category: Optional[str] = None
    session_id: Optional[str] = None

class LegalResponse(BaseModel):
    success: bool
    question: str
    response: Dict
    metadata: Dict
    session_id: str

class FeedbackRequest(BaseModel):
    conversation_id: int
    feedback: int  # 1 for positive, -1 for negative, 0 for neutral

class FAQEntry(BaseModel):
    category: str
    subcategory: Optional[str] = None
    question: str
    answer: str
    keywords: Optional[List[str]] = None

# Dependency to initialize Gemini service
def get_gemini_service():
    global gemini_service
    if gemini_service is None:
        try:
            gemini_service = GeminiLegalService()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {str(e)}")
            raise HTTPException(status_code=500, detail="AI service unavailable")
    return gemini_service

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        db.init_database()
        logger.info("Database initialized")
        
        # Test Gemini service
        gemini = get_gemini_service()
        if gemini.test_connection():
            logger.info("Gemini service connected")
        else:
            logger.error("Gemini service connection failed")
            
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Legal Assistant - Urdu Legal Guidance API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        gemini = get_gemini_service()
        db_status = "healthy"
        ai_status = "healthy" if gemini.test_connection() else "unhealthy"
        
        return {
            "status": "healthy" if db_status == "healthy" and ai_status == "healthy" else "degraded",
            "database": db_status,
            "ai_service": ai_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/legal-guidance", response_model=LegalResponse)
async def get_legal_guidance(question: LegalQuestion, gemini: GeminiLegalService = Depends(get_gemini_service)):
    """Get legal guidance for a question"""
    try:
        # Generate session ID if not provided
        session_id = question.session_id or str(uuid.uuid4())
        
        # Get legal guidance from Gemini
        response = gemini.get_legal_guidance(question.question, question.category)
        
        # Save conversation to database
        if response["success"]:
            db.save_conversation(session_id, question.question, response, question.category)
        
        # Add session ID to response
        response["session_id"] = session_id
        
        return LegalResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in legal guidance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate legal guidance")

@app.get("/api/categories")
async def get_legal_categories():
    """Get available legal categories"""
    return {
        "categories": Config.LEGAL_CATEGORIES,
        "total_categories": len(Config.LEGAL_CATEGORIES)
    }

@app.get("/api/conversation-history/{session_id}")
async def get_conversation_history(session_id: str, limit: int = 10):
    """Get conversation history for a session"""
    try:
        conversations = db.get_conversation_history(session_id, limit)
        return {
            "session_id": session_id,
            "conversations": conversations,
            "total": len(conversations)
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get conversation history")

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for a conversation"""
    try:
        success = db.update_user_feedback(feedback.conversation_id, feedback.feedback)
        if success:
            return {"message": "Feedback submitted successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to submit feedback")
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.get("/api/faq/search")
async def search_faq(query: str, category: Optional[str] = None):
    """Search FAQ database"""
    try:
        results = db.search_faq(query, category)
        return {
            "query": query,
            "category": category,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching FAQ: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search FAQ")

@app.get("/api/faq/category/{category}")
async def get_faq_by_category(category: str):
    """Get all FAQs for a category"""
    try:
        results = db.get_faq_by_category(category)
        return {
            "category": category,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Error getting FAQ by category: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get FAQ by category")

@app.post("/api/faq")
async def add_faq(faq: FAQEntry):
    """Add new FAQ entry"""
    try:
        success = db.save_faq(
            category=faq.category,
            subcategory=faq.subcategory,
            question=faq.question,
            answer=faq.answer,
            keywords=faq.keywords
        )
        if success:
            return {"message": "FAQ added successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add FAQ")
    except Exception as e:
        logger.error(f"Error adding FAQ: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add FAQ")

@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = db.get_statistics()
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.get("/api/sample-questions")
async def get_sample_questions():
    """Get sample legal questions for testing"""
    return {
        "questions": Config.SAMPLE_QUESTIONS,
        "total": len(Config.SAMPLE_QUESTIONS)
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    
    # Create .env template if it doesn't exist
    from src.config import create_env_template
    create_env_template()
    
    print("Starting Legal Assistant API Server...")
    print("Make sure to set your GEMINI_API_KEY in .env file")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 