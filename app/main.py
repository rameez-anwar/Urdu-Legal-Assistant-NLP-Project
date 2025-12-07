#!/usr/bin/env python3
"""
Main application for Urdu Legal Guidance System
Demonstrates the complete functionality with interactive CLI
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List

# Fix Urdu text display in terminal
if os.name == 'nt':  # Windows
    os.system("chcp 65001 > nul")
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Set console output to UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add parent directory and src to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.config import Config, create_env_template
from src.gemini_service import GeminiLegalService
from src.database import LegalDatabase
from src.rag_model import RAGLegalAssistant
from app.api_server import app
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrduLegalGuide:
    """Main application class for Urdu Legal Guidance System"""
    
    def __init__(self):
        """Initialize the application"""
        self.gemini_service = None
        self.rag_model = None
        self.db = None
        self.session_id = None
        
    def setup(self):
        """Setup the application"""
        print("🔧 Setting up Legal Assistant - Urdu Legal Guidance System...")
        
        # Create .env template if needed
        create_env_template()
        
        # Check for API key
        if not Config.GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY not found in .env file")
            print("🔑 Please add your Google Gemini API key to .env file")
            print("📝 You can get it from: https://makersuite.google.com/app/apikey")
            return False
        
        try:
            # Initialize database first
            self.db = LegalDatabase()
            
            # Try to initialize RAG model (if index exists)
            try:
                self.rag_model = RAGLegalAssistant()
                self.gemini_service = GeminiLegalService(db=self.db, rag_model=self.rag_model)
                print("✅ RAG model initialized (with document retrieval)")
            except Exception as e:
                print(f"⚠️  RAG model not available: {str(e)}")
                print("📝 Using FAQ-based mode only")
                self.gemini_service = GeminiLegalService(db=self.db)
            
            # Test connections
            if not self.gemini_service.test_connection():
                print("❌ Failed to connect to Gemini API")
                return False
            
            print("✅ All services initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            return False
    
    def interactive_mode(self):
        """Run interactive CLI mode"""
        print("\n" + "="*60)
        print("🏛️  Legal Assistant - Roman Urdu Legal Guidance System")
        print("="*60)
        print("💡 Type your legal questions in Roman Urdu")
        print("📝 Type 'help' for commands, 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 Your question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'categories':
                    self.show_categories()
                    continue
                elif user_input.lower() == 'samples':
                    self.show_sample_questions()
                    continue
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_statistics()
                    continue
                elif user_input.lower() == 'faq':
                    self.search_faq_mode()
                    continue
                
                # Process legal question
                self.process_legal_question(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def process_legal_question(self, question: str, category: str = None):
        """Process a legal question and display response"""
        try:
            print("🤖 Processing your question...")
            
            # Get legal guidance
            response = self.gemini_service.get_legal_guidance(question, category)
            
            if response["success"]:
                # Save to database
                if self.session_id:
                    self.db.save_conversation(self.session_id, question, response, category)
                
                # Display response
                self.display_response(response)
            else:
                print(f"❌ Error: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error processing question: {str(e)}")
    
    def display_response(self, response: Dict):
        """Display formatted response"""
        print("\n" + "📋" + "="*50)
        print("📝 LEGAL GUIDANCE")
        print("="*50)
        
        # Question
        print(f"❓ Question: {response['question']}")
        print()
        
        # Explanation
        if response['response']['explanation']:
            print("💡 Explanation:")
            print("-" * 30)
            explanation = response['response']['explanation'].strip()
            # Try to display in Urdu, fallback to English if needed
            try:
                print(explanation)
            except UnicodeEncodeError:
                print("(Urdu text - see API response for full text)")
            print()
        
        # Practical steps
        if response['response']['practical_steps']:
            print("📋 Practical Steps:")
            print("-" * 30)
            steps = response['response']['practical_steps'].strip()
            try:
                print(steps)
            except UnicodeEncodeError:
                print("(Urdu text - see API response for full text)")
            print()
        
        # When to consult lawyer
        if response['response']['when_to_consult_lawyer']:
            print("⚖️  When to Consult a Lawyer:")
            print("-" * 30)
            lawyer_advice = response['response']['when_to_consult_lawyer'].strip()
            try:
                print(lawyer_advice)
            except UnicodeEncodeError:
                print("(Urdu text - see API response for full text)")
            print()
        
        # Disclaimers
        print("⚠️  Disclaimer:")
        print("-" * 30)
        disclaimer = response['response']['disclaimers'].strip()
        try:
            print(disclaimer)
        except UnicodeEncodeError:
            print("This information is for general guidance only, not legal advice.")
        print("="*50)
    
    def show_help(self):
        """Show help information"""
        print("\n📚 Available Commands:")
        print("  help      - Show this help")
        print("  categories - Show legal categories")
        print("  samples   - Show sample questions")
        print("  history   - Show conversation history")
        print("  stats     - Show system statistics")
        print("  faq       - Search FAQ database")
        print("  quit      - Exit the application")
        print("\n💡 Just type your legal question to get guidance!")
    
    def show_categories(self):
        """Show available legal categories"""
        print("\n📂 Legal Categories:")
        for key, value in Config.LEGAL_CATEGORIES.items():
            print(f"  • {value['name']} ({key})")
            for subcategory in value['subcategories']:
                print(f"    - {subcategory}")
        print()
    
    def show_sample_questions(self):
        """Show sample questions"""
        print("\n📝 Sample Questions (Roman Urdu):")
        sample_questions = Config.SAMPLE_QUESTIONS
        for i, question in enumerate(sample_questions, 1):
            print(f"  {i}. {question}")
        print()
    
    def show_conversation_history(self):
        """Show conversation history"""
        if not self.session_id:
            print("❌ No active session")
            return
        
        conversations = self.db.get_conversation_history(self.session_id)
        if not conversations:
            print("📝 No conversation history found")
            return
        
        print(f"\n📚 Conversation History (Session: {self.session_id})")
        for i, conv in enumerate(conversations, 1):
            print(f"\n  {i}. Question: {conv['question']}")
            print(f"     Answer: {conv['answer']['response']['explanation'][:100]}...")
            print(f"     Time: {conv['timestamp']}")
    
    def show_statistics(self):
        """Show system statistics"""
        stats = self.db.get_statistics()
        print("\n📊 System Statistics:")
        print(f"  Total Conversations: {stats.get('total_conversations', 0)}")
        print(f"  Total FAQ Entries: {stats.get('total_faq', 0)}")
        print(f"  Active Sessions: {stats.get('active_sessions', 0)}")
        
        if stats.get('category_distribution'):
            print("  Category Distribution:")
            for category, count in stats['category_distribution'].items():
                print(f"    {category}: {count}")
        print()
    
    def search_faq_mode(self):
        """Interactive FAQ search mode"""
        print("\n🔍 FAQ Search Mode")
        print("Type 'back' to return to main menu")
        
        while True:
            query = input("\n🔍 Search FAQ: ").strip()
            
            if query.lower() == 'back':
                break
            
            if not query:
                continue
            
            results = self.db.search_faq(query)
            
            if results:
                print(f"\n📚 Found {len(results)} FAQ entries:")
                for i, result in enumerate(results, 1):
                    print(f"\n  {i}. Category: {result['category']}")
                    print(f"     Question: {result['question']}")
                    print(f"     Answer: {result['answer'][:100]}...")
            else:
                print("❌ No FAQ entries found")
    
    def demo_mode(self):
        """Run demo with sample questions"""
        print("\n🎬 Running Demo Mode...")
        print("This will process sample legal questions to demonstrate the system.")
        
        sample_questions = [
            ("Mere shohar ne talaq ka notice bheja hai, main kya karun?", "family_law"),
            ("Police mein FIR darj karane ka tarika kya hai?", "criminal_law"),
            ("Jaidat ki taqseem ke liye kya karna chahiye?", "property_law")
        ]
        
        for question, category in sample_questions:
            print(f"\n{'='*60}")
            print(f"Demo Question: {question}")
            print(f"Category: {category}")
            print("="*60)
            
            self.process_legal_question(question, category)
            
            # Pause between questions
            input("\nPress Enter to continue...")
        
        print("\n✅ Demo completed!")
    
    def start_api_server(self):
        """Start the FastAPI server"""
        print("🚀 Starting API Server...")
        print("📡 API will be available at: http://localhost:8000")
        print("📚 API documentation at: http://localhost:8000/docs")
        print("🔧 Press Ctrl+C to stop the server")
        
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )

def main():
    """Main entry point"""
    print("Legal Assistant - Roman Urdu Legal Guidance System")
    print("="*50)
    
    # Initialize application
    app = UrduLegalGuide()
    
    # Setup
    if not app.setup():
        print("Setup failed. Please check your configuration.")
        return
    
    # Show menu
    print("\nChoose an option:")
    print("1. Interactive Mode (CLI)")
    print("2. Demo Mode (Sample Questions)")
    print("3. Start API Server")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                app.interactive_mode()
                break
            elif choice == "2":
                app.demo_mode()
                break
            elif choice == "3":
                app.start_api_server()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 