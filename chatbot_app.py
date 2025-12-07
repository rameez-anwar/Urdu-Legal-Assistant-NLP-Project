#!/usr/bin/env python3
"""
Beautiful Chatbot Web Application for Legal Assistant
Roman Urdu Legal Guidance System
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import uuid

# Add current directory and src to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.config import Config, create_env_template
from src.gemini_service import GeminiLegalService
from src.database import LegalDatabase
from src.rag_model import RAGLegalAssistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'legal-assistant-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize services
gemini_service = None
rag_model = None
db = LegalDatabase()

def initialize_services():
    """Initialize AI and database services"""
    global gemini_service, rag_model
    
    try:
        # Create .env template if needed
        create_env_template()
        
        # Check for API key
        if not Config.GEMINI_API_KEY:
            return False, "GEMINI_API_KEY not found in .env file"
        
        # Initialize database first
        db.init_database()
        
        # Try to initialize RAG model (if index exists)
        try:
            rag_model = RAGLegalAssistant()
            gemini_service = GeminiLegalService(db=db, rag_model=rag_model)
            logger.info("✅ RAG model initialized (with document retrieval)")
        except Exception as e:
            logger.warning(f"⚠️  RAG model not available: {str(e)}")
            logger.info("📝 Using FAQ-based mode only")
            gemini_service = GeminiLegalService(db=db)
        
        # Test connection
        if not gemini_service.test_connection():
            return False, "Failed to connect to Gemini API"
        
        return True, "Services initialized successfully"
        
    except Exception as e:
        return False, f"Setup failed: {str(e)}"

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get legal guidance
        response = gemini_service.get_legal_guidance(message)
        
        if response["success"]:
            # Save to database
            db.save_conversation(session_id, message, response)
            
            # Format response for chat
            chat_response = format_chat_response(response)
            
            return jsonify({
                'success': True,
                'response': chat_response,
                'session_id': session_id
            })
        else:
            return jsonify({
                'success': False,
                'error': response.get('error', 'Unknown error')
            }), 500
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def format_chat_response(response):
    """Format response for chat display"""
    formatted = {
        'question': response['question'],
        'explanation': response['response']['explanation'],
        'practical_steps': response['response']['practical_steps'],
        'when_to_consult_lawyer': response['response']['when_to_consult_lawyer'],
        'disclaimers': response['response']['disclaimers']
    }
    return formatted

@app.route('/api/categories')
def get_categories():
    """Get legal categories"""
    return jsonify({
        'categories': Config.LEGAL_CATEGORIES
    })

@app.route('/api/sample-questions')
def get_sample_questions():
    """Get sample questions"""
    sample_questions = [
        "Mere shohar ne talaq ka notice bheja hai, main kya karun?",
        "Police mein FIR darj karane ka tarika kya hai?",
        "Jaidat ki taqseem ke liye kya karna chahiye?",
        "Mere pass warisat ka masla hai, main kya karun?",
        "Tenant ko kese nikalein?"
    ]
    return jsonify({'questions': sample_questions})

@app.route('/api/faq/search')
def search_faq():
    """Search FAQ database"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        results = db.search_faq(query)
        return jsonify({
            'query': query,
            'results': results,
            'total': len(results)
        })
    except Exception as e:
        logger.error(f"FAQ search error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/faq/top')
def get_top_faqs():
    """Get top used FAQs"""
    try:
        limit = request.args.get('limit', 5, type=int)
        results = db.get_top_faqs(limit)
        return jsonify({
            'results': results,
            'total': len(results)
        })
    except Exception as e:
        logger.error(f"Top FAQ error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        if gemini_service and gemini_service.test_connection():
            # Get database stats
            stats = db.get_statistics()
            return jsonify({
                'status': 'healthy', 
                'message': 'All services running',
                'database_stats': stats
            })
        else:
            return jsonify({'status': 'unhealthy', 'message': 'AI service unavailable'}), 500
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'message': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    # Connection established silently
    pass

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    # Initialize services
    success, message = initialize_services()
    
    if success:
        print("✅ Services initialized successfully!")
        print("🚀 Starting Legal Assistant Chatbot...")
        print("📱 Open your browser and go to: http://localhost:5000")
        
        # Run the app
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    else:
        print(f"❌ Failed to initialize services: {message}")
        print("🔑 Please check your .env file and API key") 