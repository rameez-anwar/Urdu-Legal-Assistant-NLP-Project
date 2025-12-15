import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Google Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash" 
    
    # Application Settings
    APP_NAME = "Legal Assistant - Roman Urdu Legal Guide"
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Database Settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./urdu_legal.db")
    DATABASE_PATH = "./urdu_legal.db"
    
    # Legal Categories (Roman Urdu)
    LEGAL_CATEGORIES = {
        "family_law": {
            "name": "Family Law (Khandani Qanoon)",
            "subcategories": [
                "Talaq aur Khula",
                "Nikah aur Shadi",
                "Bachon ki custody",
                "Mehr aur Jahez",
                "Warisi"
            ]
        },
        "criminal_law": {
            "name": "Criminal Law (Jurm Qanoon)",
            "subcategories": [
                "FIR darj karana",
                "Police investigation",
                "Bail aur warrant",
                "Court proceedings",
                "Jail aur punishment"
            ]
        },
        "property_law": {
            "name": "Property Law (Jaidat Qanoon)",
            "subcategories": [
                "Jaidat ki taqseem",
                "Rent aur lease",
                "Land disputes",
                "Property documents",
                "Encroachment"
            ]
        },
        "civil_law": {
            "name": "Civil Law (Madani Qanoon)",
            "subcategories": [
                "Contract disputes",
                "Money recovery",
                "Defamation",
                "Consumer rights",
                "Employment issues"
            ]
        },
        "constitutional_law": {
            "name": "Constitutional Law (Dastoor Qanoon)",
            "subcategories": [
                "Fundamental rights",
                "Election issues",
                "Government services",
                "Discrimination",
                "Freedom of speech"
            ]
        }
    }
    
    # Sample Questions (Roman Urdu)
    SAMPLE_QUESTIONS = [
        "Mere shohar ne talaq ka notice bheja hai, main kya karun?",
        "Police mein FIR darj karane ka tarika kya hai?",
        "Jaidat ki taqseem ke liye kya karna chahiye?",
        "Mere pass warisat ka masla hai, main kya karun?",
        "Tenant ko kese nikalein?",
        "Bail lene ka tarika kya hai?",
        "Court mein case file karne ka process kya hai?",
        "Mehr ki raqam wapis lene ka tarika?",
        "Land dispute mein kya karna chahiye?",
        "Police harassment ke khilaf kya karna chahiye?"
    ]
    
    # Prompt Templates (Roman Urdu)
    LEGAL_PROMPT_TEMPLATE = """
Aap Pakistan ke legal system ke expert hain. User ka sawal: {user_question}

Aap ko Roman Urdu mein jawab dena hai with this format:

**Explanation:**
[Roman Urdu mein detailed explanation]

**Practical Steps:**
[Roman Urdu mein step-by-step guide]

**When to Consult a Lawyer:**
[Roman Urdu mein guidance]

**Disclaimers:**
Ye information general guidance ke liye hai, legal advice nahi.

Important:
1. Sirf Roman Urdu use karein
2. Simple language use karein
3. Practical advice dein
4. Safety emphasize karein
"""
    
    # Disclaimers (Roman Urdu)
    DISCLAIMERS = {
        "roman_urdu": "Ye information general guidance ke liye hai, legal advice nahi. Complex cases mein lawyer se consult karein.",
        "english": "This information is for general guidance only, not legal advice. Consult a lawyer for complex cases."
    }
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    CHAT_PORT = 5000
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def get_current_timestamp(cls):
        """Get current timestamp"""
        return datetime.now()
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            return False, "GEMINI_API_KEY not found"
        return True, "Configuration valid"

def create_env_template():
    """Create .env template if it doesn't exist"""
    env_file = ".env"
    if not os.path.exists(env_file):
        env_content = """# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Application Settings
DEBUG=True
APP_NAME=Legal Assistant - Roman Urdu Legal Guide

# Database Settings
DATABASE_URL=sqlite:///./urdu_legal.db
"""
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(env_content)
        print(".env file created!")
        print("Please add your Gemini API key to .env file")
    else:
        print(".env file already exists")

# Test configuration
if __name__ == "__main__":
    print("Testing Legal Assistant Configuration...")
    
    # Create .env template
    create_env_template()
    
    # Validate config
    is_valid, message = Config.validate_config()
    
    if is_valid:
        print("Configuration is valid!")
        print(f"App Name: {Config.APP_NAME}")
        print(f"Model: {Config.GEMINI_MODEL}")
        print(f"Categories: {len(Config.LEGAL_CATEGORIES)}")
        print(f"Sample Questions: {len(Config.SAMPLE_QUESTIONS)}")
    else:
        print(f"Configuration error: {message}")
        print("Please add your GEMINI_API_KEY to .env file") 