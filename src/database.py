import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDatabase:
    """Database class for legal guidance system"""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        self.db_path = db_path or Config.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        category TEXT,
                        feedback INTEGER DEFAULT 0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create FAQ table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS faqs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        keywords TEXT,
                        usage_count INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create user_sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create legal_documents table for RAG
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS legal_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        category TEXT,
                        segment_id INTEGER,
                        text TEXT NOT NULL,
                        embedding BLOB,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create index for faster searches
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_category ON legal_documents(category)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_filename ON legal_documents(filename)
                ''')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
                # Initialize FAQ data if table is empty
                self._initialize_faq_data()
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def _initialize_faq_data(self):
        """Initialize FAQ data with common legal questions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if FAQ table is empty
                cursor.execute("SELECT COUNT(*) FROM faqs")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    # Insert initial FAQ data
                    faq_data = [
                        {
                            "category": "family_law",
                            "subcategory": "Talaq aur Khula",
                            "question": "Mere shohar ne talaq ka notice bheja hai, main kya karun?",
                            "answer": "Agar aap ke shohar ne talaq ka notice bheja hai to aap ye kar sakte hain:\n\n1. Notice ko carefully read karein\n2. Apne legal rights ke bare mein janein\n3. Family lawyer se consult karein\n4. Court mein response file karein\n5. Bachon ki custody ke liye application dein\n\nImportant: Talaq ke 90 din ka period hota hai. Is time mein reconciliation possible hai.",
                            "keywords": "talaq, divorce, notice, shohar, wife, court, lawyer"
                        },
                        {
                            "category": "criminal_law",
                            "subcategory": "FIR darj karana",
                            "question": "Police mein FIR darj karane ka tarika kya hai?",
                            "answer": "FIR darj karane ke liye ye steps follow karein:\n\n1. Police station mein jaein\n2. Complaint in writing dein\n3. Apna CNIC aur documents sath lein\n4. FIR copy zaroor lein\n5. Case number note karein\n\nNote: Police FIR darj karne se mana nahi kar sakti. Agar mana kare to SP ya DSP se contact karein.",
                            "keywords": "FIR, police, complaint, criminal, case, investigation"
                        },
                        {
                            "category": "property_law",
                            "subcategory": "Jaidat ki taqseem",
                            "question": "Jaidat ki taqseem ke liye kya karna chahiye?",
                            "answer": "Jaidat ki taqseem ke liye ye process follow karein:\n\n1. Property documents collect karein\n2. Family members ka agreement banayein\n3. Lawyer se legal advice lein\n4. Court mein partition suit file karein\n5. Property valuation karayein\n\nImportant: Verbal agreements valid nahi hain. Written agreement zaroori hai.",
                            "keywords": "jaidat, property, taqseem, partition, court, documents"
                        },
                        {
                            "category": "family_law",
                            "subcategory": "Warisat",
                            "question": "Mere pass warisat ka masla hai, main kya karun?",
                            "answer": "Warisat ke masle mein ye steps follow karein:\n\n1. Original documents collect karein\n2. Family tree banayein\n3. Succession certificate ke liye apply karein\n4. Court mein case file karein\n5. Legal heirs ka list banayein\n\nNote: Islamic law mein warisat ka specific process hai. Lawyer se consult karein.",
                            "keywords": "warisat, inheritance, property, legal heirs, court, documents"
                        },
                        {
                            "category": "property_law",
                            "subcategory": "Rent aur lease",
                            "question": "Tenant ko kese nikalein?",
                            "answer": "Tenant ko legally nikalan ke liye:\n\n1. Legal notice dein (30 days)\n2. Rent arrears clear karein\n3. Court mein eviction suit file karein\n4. Proper evidence collect karein\n5. Court order ka wait karein\n\nWarning: Illegal eviction crime hai. Sirf legal process follow karein.",
                            "keywords": "tenant, eviction, rent, lease, court, notice"
                        },
                        {
                            "category": "criminal_law",
                            "subcategory": "Bail aur warrant",
                            "question": "Bail lene ka tarika kya hai?",
                            "answer": "Bail lene ke liye ye process follow karein:\n\n1. Lawyer hire karein\n2. Bail application prepare karein\n3. Court mein file karein\n4. Surety arrange karein\n5. Court hearing attend karein\n\nImportant: Bail right nahi, privilege hai. Court discretion use karti hai.",
                            "keywords": "bail, warrant, court, lawyer, arrest, criminal"
                        },
                        {
                            "category": "civil_law",
                            "subcategory": "Contract disputes",
                            "question": "Contract breach ke case mein kya karna chahiye?",
                            "answer": "Contract breach ke case mein:\n\n1. Contract copy collect karein\n2. Breach evidence collect karein\n3. Legal notice dein\n4. Court mein suit file karein\n5. Damages claim karein\n\nNote: Oral contracts prove karna mushkil hai. Written contracts zaroori hain.",
                            "keywords": "contract, breach, court, damages, legal notice"
                        },
                        {
                            "category": "constitutional_law",
                            "subcategory": "Fundamental rights",
                            "question": "Police harassment ke khilaf kya karna chahiye?",
                            "answer": "Police harassment ke khilaf ye steps follow karein:\n\n1. Incident record karein\n2. Witnesses collect karein\n3. SP/IG ko complaint dein\n4. Human Rights Commission ko contact karein\n5. Court mein petition file karein\n\nImportant: Harassment report karna aap ka right hai.",
                            "keywords": "police, harassment, rights, complaint, court, human rights"
                        }
                    ]
                    
                    for faq in faq_data:
                        cursor.execute('''
                            INSERT INTO faqs (category, subcategory, question, answer, keywords)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            faq["category"],
                            faq["subcategory"],
                            faq["question"],
                            faq["answer"],
                            faq["keywords"]
                        ))
                    
                    conn.commit()
                    logger.info(f"Initialized {len(faq_data)} FAQ entries")
                
        except Exception as e:
            logger.error(f"FAQ initialization failed: {str(e)}")
    
    def save_conversation(self, session_id: str, question: str, response: Dict, category: str = None) -> bool:
        """Save conversation to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save or update session
                cursor.execute('''
                    INSERT OR REPLACE INTO user_sessions (session_id, last_activity)
                    VALUES (?, CURRENT_TIMESTAMP)
                ''', (session_id,))
                
                # Save conversation
                cursor.execute('''
                    INSERT INTO conversations (session_id, question, answer, category)
                    VALUES (?, ?, ?, ?)
                ''', (
                    session_id,
                    question,
                    json.dumps(response, ensure_ascii=False),
                    category
                ))
                
                conn.commit()
                logger.info(f"Conversation saved for session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT question, answer, category, timestamp
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (session_id, limit))
                
                conversations = []
                for row in cursor.fetchall():
                    conversations.append({
                        'question': row[0],
                        'answer': json.loads(row[1]) if row[1] else {},
                        'category': row[2],
                        'timestamp': row[3]
                    })
                
                return conversations
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            return []
    
    def search_faq(self, query: str, category: str = None) -> List[Dict]:
        """Search FAQ database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update usage count for searched terms
                cursor.execute('''
                    UPDATE faqs 
                    SET usage_count = usage_count + 1
                    WHERE question LIKE ? OR keywords LIKE ?
                ''', (f'%{query}%', f'%{query}%'))
                
                # Search query
                if category:
                    cursor.execute('''
                        SELECT category, subcategory, question, answer, usage_count
                        FROM faqs
                        WHERE category = ? AND (question LIKE ? OR keywords LIKE ?)
                        ORDER BY usage_count DESC, created_at DESC
                    ''', (category, f'%{query}%', f'%{query}%'))
                else:
                    cursor.execute('''
                        SELECT category, subcategory, question, answer, usage_count
                        FROM faqs
                        WHERE question LIKE ? OR keywords LIKE ?
                        ORDER BY usage_count DESC, created_at DESC
                    ''', (f'%{query}%', f'%{query}%'))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'category': row[0],
                        'subcategory': row[1],
                        'question': row[2],
                        'answer': row[3],
                        'usage_count': row[4]
                    })
                
                conn.commit()
                return results
                
        except Exception as e:
            logger.error(f"Failed to search FAQ: {str(e)}")
            return []
    
    def get_faq_by_category(self, category: str) -> List[Dict]:
        """Get all FAQs for a category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT category, subcategory, question, answer, usage_count
                    FROM faqs
                    WHERE category = ?
                    ORDER BY usage_count DESC, created_at DESC
                ''', (category,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'category': row[0],
                        'subcategory': row[1],
                        'question': row[2],
                        'answer': row[3],
                        'usage_count': row[4]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get FAQ by category: {str(e)}")
            return []
    
    def save_faq(self, category: str, subcategory: str, question: str, answer: str, keywords: str = None) -> bool:
        """Save new FAQ entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO faqs (category, subcategory, question, answer, keywords)
                    VALUES (?, ?, ?, ?, ?)
                ''', (category, subcategory, question, answer, keywords))
                
                conn.commit()
                logger.info(f"FAQ saved: {question[:50]}...")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save FAQ: {str(e)}")
            return False
    
    def update_user_feedback(self, conversation_id: int, feedback: int) -> bool:
        """Update user feedback for a conversation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE conversations
                    SET feedback = ?
                    WHERE id = ?
                ''', (feedback, conversation_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update feedback: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total conversations
                cursor.execute("SELECT COUNT(*) FROM conversations")
                total_conversations = cursor.fetchone()[0]
                
                # Get total FAQ entries
                cursor.execute("SELECT COUNT(*) FROM faqs")
                total_faq = cursor.fetchone()[0]
                
                # Get active sessions
                cursor.execute("SELECT COUNT(*) FROM user_sessions")
                active_sessions = cursor.fetchone()[0]
                
                # Get category distribution
                cursor.execute('''
                    SELECT category, COUNT(*) 
                    FROM conversations 
                    WHERE category IS NOT NULL 
                    GROUP BY category
                ''')
                category_distribution = dict(cursor.fetchall())
                
                return {
                    'total_conversations': total_conversations,
                    'total_faq': total_faq,
                    'active_sessions': active_sessions,
                    'category_distribution': category_distribution
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
    
    def get_top_faqs(self, limit: int = 5) -> List[Dict]:
        """Get most used FAQs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT category, subcategory, question, answer, usage_count
                    FROM faqs
                    ORDER BY usage_count DESC
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'category': row[0],
                        'subcategory': row[1],
                        'question': row[2],
                        'answer': row[3],
                        'usage_count': row[4]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get top FAQs: {str(e)}")
            return [] 