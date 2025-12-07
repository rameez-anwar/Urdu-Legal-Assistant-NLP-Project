#!/usr/bin/env python3
"""
Google Gemini API Service for Legal Assistant
Handles AI interactions for legal guidance in both Urdu script and Roman Urdu
"""

import os
import sys
import json
import logging
import re
from typing import Dict, List, Optional
import google.generativeai as genai
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLegalService:
    """Service class for Google Gemini AI legal guidance"""
    
    def __init__(self, db=None, rag_model=None):
        """Initialize the Gemini service"""
        try:
            # Configure Gemini
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            self.chat = None
            self.db = db
            self.rag_model = rag_model  # Optional RAG model for enhanced responses
            logger.info("✅ Gemini service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini service: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API"""
        try:
            response = self.model.generate_content("Hello")
            return True
        except Exception as e:
            logger.error(f"❌ Gemini connection test failed: {str(e)}")
            return False
    
    def get_legal_guidance(self, question: str, category: str = None) -> Dict:
        """Get legal guidance for a question in the same language as input"""
        try:
            # First check if this is a legal question
            if not self._is_legal_question(question):
                return self._get_off_topic_response(question)
            
            # Detect input language
            is_roman_urdu = self._is_roman_urdu(question)
            
            # Use RAG if available (preferred method)
            if self.rag_model:
                logger.info("Using RAG model for enhanced response")
                return self.rag_model.generate_response(question)
            
            # Search FAQ database first
            faq_results = self._search_faq_database(question)
            
            # If FAQ found, use it as primary source and enhance with AI
            if faq_results:
                return self._get_enhanced_faq_response(faq_results[0], question, is_roman_urdu)
            
            # If no FAQ found, use pure AI response
            prompt = self._build_legal_prompt(question, category, is_roman_urdu, [])
            response = self.model.generate_content(prompt)
            parsed_response = self._parse_legal_response(response.text, question, is_roman_urdu)
            
            return {
                "success": True,
                "question": question,
                "response": parsed_response,
                "metadata": {
                    "category": category,
                    "model": Config.GEMINI_MODEL,
                    "language": "roman_urdu" if is_roman_urdu else "urdu",
                    "timestamp": str(Config.get_current_timestamp()),
                    "faq_used": False,
                    "response_type": "pure_ai"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting legal guidance: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
    
    def _is_legal_question(self, question: str) -> bool:
        """Check if the question is legal-related"""
        legal_keywords = [
            # Roman Urdu legal terms
            "talaq", "divorce", "shohar", "wife", "court", "lawyer", "wakeel",
            "police", "FIR", "case", "jaidat", "property", "warisat", "inheritance",
            "tenant", "rent", "lease", "bail", "warrant", "contract", "breach",
            "harassment", "rights", "complaint", "notice", "document", "agreement",
            "custody", "mehr", "jahez", "nikah", "shadi", "eviction", "damages",
            "suit", "petition", "hearing", "judgment", "appeal", "settlement",
            
            # Urdu script legal terms
            "طلاق", "شوہر", "بیوی", "عدالت", "وکیل", "پولیس", "جائیداد", "وراثت",
            "کرایہ", "ضمانت", "معاہدہ", "شکایت", "نوٹس", "دستاویز", "اتفاق",
            "تحویل", "مہر", "جہیز", "نکاح", "شادی", "خارج", "نقصان", "مقدمہ",
            "درخواست", "سماعت", "فیصلہ", "اپیل", "تصفیہ"
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in legal_keywords)
    
    def _get_off_topic_response(self, question: str) -> Dict:
        """Generate response for off-topic questions"""
        is_roman_urdu = self._is_roman_urdu(question)
        
        if is_roman_urdu:
            message = "Maaf kijiye, main sirf legal guidance ke liye banaya gaya hun. Aap ka sawal legal domain mein nahi aata. Agar aap ko koi legal masla hai to zaroor poochein."
        else:
            message = "معذرت، میں صرف قانونی رہنمائی کے لیے بنایا گیا ہوں۔ آپ کا سوال قانونی شعبے میں نہیں آتا۔ اگر آپ کو کوئی قانونی مسئلہ ہے تو ضرور پوچھیں۔"
        
        return {
            "success": True,
            "question": question,
            "response": {
                "explanation": message,
                "practical_steps": "",
                "when_to_consult_lawyer": "",
                "disclaimers": "Ye information general guidance ke liye hai, legal advice nahi." if is_roman_urdu 
                              else "یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔"
            },
            "metadata": {
                "category": "off_topic",
                "model": Config.GEMINI_MODEL,
                "language": "roman_urdu" if is_roman_urdu else "urdu",
                "timestamp": str(Config.get_current_timestamp())
            }
        }
    
    def _search_faq_database(self, question: str) -> List[Dict]:
        """Search FAQ database for relevant answers"""
        if not self.db:
            return []
        
        try:
            # Search in FAQ database
            faq_results = self.db.search_faq(question)
            return faq_results
        except Exception as e:
            logger.error(f"❌ FAQ search failed: {str(e)}")
            return []
    
    def _is_roman_urdu(self, text: str) -> bool:
        """Detect if text is in Roman Urdu or Urdu script"""
        # Check for Urdu script characters (Arabic/Persian Unicode range)
        urdu_chars = 0
        roman_chars = 0
        
        for char in text:
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                urdu_chars += 1
            elif char.isalpha() and char.isascii():
                roman_chars += 1
        
        # If more Urdu script characters, it's Urdu script
        if urdu_chars > roman_chars:
            return False
        else:
            return True
    
    def _build_legal_prompt(self, question: str, category: str = None, is_roman_urdu: bool = True, faq_results: List[Dict] = None) -> str:
        """Build the legal guidance prompt in the appropriate language"""
        
        category_info = ""
        if category and category in Config.LEGAL_CATEGORIES:
            category_info = f"\nCategory: {Config.LEGAL_CATEGORIES[category]['name']}"
        
        # Add FAQ context if available
        faq_context = ""
        if faq_results:
            faq_context = "\n\nRelevant FAQ Information:\n"
            for faq in faq_results[:2]:  # Use top 2 FAQ results
                faq_context += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
        
        if is_roman_urdu:
            # Roman Urdu prompt
            prompt = f"""Aap Pakistan ke legal system ke expert hain. Aap ko Roman Urdu mein jawab dena hai.

Question: {question}{category_info}{faq_context}

Aap ko ye format mein jawab dena hai:

**Explanation:**
[Roman Urdu mein detailed explanation]

**Practical Steps:**
[Roman Urdu mein step-by-step guide]

**When to Consult a Lawyer:**
[Roman Urdu mein guidance]

**Disclaimers:**
Ye information general guidance ke liye hai, legal advice nahi. Complex cases mein lawyer se consult karein.

Important Instructions:
1. Sirf Roman Urdu use karein (Urdu script nahi)
2. Simple aur samajhne mein aasan language use karein
3. Practical aur actionable advice dein
4. Legal jargon avoid karein
5. Safety aur legal rights emphasize karein
6. Har section mein 2-3 points dein maximum
7. FAQ data ko integrate karein agar relevant ho

Example Roman Urdu words:
- Talaq = Divorce
- FIR = First Information Report
- Jaidat = Property
- Warisat = Inheritance
- Qanoon = Law
- Adalat = Court
- Wakeel = Lawyer
- Police = Police
- Notice = Notice
- Document = Document

Please provide a helpful, practical response in Roman Urdu."""
        else:
            # Urdu script prompt
            prompt = f"""آپ پاکستان کے قانونی نظام کے ماہر ہیں۔ آپ کو اردو میں جواب دینا ہے۔

سوال: {question}{category_info}{faq_context}

آپ کو یہ فارمیٹ میں جواب دینا ہے:

**وضاحت:**
[اردو میں تفصیلی وضاحت]

**عملی اقدامات:**
[اردو میں مرحلہ وار رہنمائی]

**وکیل سے مشورہ کب کریں:**
[اردو میں رہنمائی]

**تنبیہ:**
یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔ پیچیدہ معاملات میں وکیل سے مشورہ کریں۔

اہم ہدایات:
1. صرف اردو استعمال کریں
2. سادہ اور سمجھنے میں آسان زبان استعمال کریں
3. عملی اور قابل عمل مشورہ دیں
4. قانونی اصطلاحات سے بچیں
5. حفاظت اور قانونی حقوق پر زور دیں
6. ہر حصے میں زیادہ سے زیادہ 2-3 نکات دیں
7. FAQ data ko integrate karein agar relevant ho

براہ کرم اردو میں مددگار اور عملی جواب دیں۔"""

        return prompt
    
    def _parse_legal_response(self, response_text: str, original_question: str, is_roman_urdu: bool = True) -> Dict:
        """Parse the AI response into structured sections"""
        try:
            # Extract sections using the _extract_sections method
            sections = self._extract_sections(response_text, is_roman_urdu)
            
            return {
                "explanation": sections.get("explanation", ""),
                "practical_steps": sections.get("practical_steps", ""),
                "when_to_consult_lawyer": sections.get("when_to_consult_lawyer", ""),
                "disclaimers": sections.get("disclaimers", 
                    "Ye information general guidance ke liye hai, legal advice nahi." if is_roman_urdu 
                    else "یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔")
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing response: {str(e)}")
            # Return the full response as explanation if parsing fails
            return {
                "explanation": response_text,
                "practical_steps": "",
                "when_to_consult_lawyer": "",
                "disclaimers": "Ye information general guidance ke liye hai, legal advice nahi." if is_roman_urdu 
                              else "یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔"
            }
    
    def _extract_sections(self, text: str, is_roman_urdu: bool = True) -> Dict:
        """Extract different sections from the response text"""
        sections = {
            "explanation": "",
            "practical_steps": "",
            "when_to_consult_lawyer": "",
            "disclaimers": ""
        }
        
        # Define section markers based on language (more flexible matching)
        if is_roman_urdu:
            explanation_markers = ["**explanation:**", "explanation:", "explanation"]
            steps_markers = ["**practical steps:**", "practical steps:", "practical steps", "steps"]
            lawyer_markers = ["**when to consult a lawyer:**", "when to consult a lawyer:", 
                            "when to consult", "consult a lawyer", "lawyer"]
            disclaimer_markers = ["**disclaimers:**", "disclaimers:", "disclaimer", "disclaimers"]
        else:
            explanation_markers = ["**وضاحت:**", "وضاحت:", "**explanation:**", "explanation:"]
            steps_markers = ["**عملی اقدامات:**", "عملی اقدامات:", "**practical steps:**", "practical steps:"]
            lawyer_markers = ["**وکیل سے مشورہ کب کریں:**", "وکیل سے مشورہ کب کریں:", "**when to consult a lawyer:**", "when to consult a lawyer:"]
            disclaimer_markers = ["**تنبیہ:**", "تنبیہ:", "**disclaimers:**", "disclaimers:"]
        
        # First, try to split by section markers if they exist
        # Look for patterns like "**Explanation:**" or "Explanation:" at start of line or paragraph
        section_patterns = []
        if is_roman_urdu:
            section_patterns = [
                (r'\n\s*\*\*Explanation:\*\*\s*\n', '\n|||SECTION:explanation|||\n'),
                (r'\n\s*Explanation\s*:\s*\n', '\n|||SECTION:explanation|||\n'),
                (r'^\*\*Explanation:\*\*\s*', '|||SECTION:explanation|||\n'),
                (r'^Explanation\s*:\s*', '|||SECTION:explanation|||\n'),
                (r'\n\s*\*\*Practical Steps:\*\*\s*\n', '\n|||SECTION:practical_steps|||\n'),
                (r'\n\s*Practical Steps\s*:\s*\n', '\n|||SECTION:practical_steps|||\n'),
                (r'^\*\*Practical Steps:\*\*\s*', '|||SECTION:practical_steps|||\n'),
                (r'^Practical Steps\s*:\s*', '|||SECTION:practical_steps|||\n'),
                (r'\n\s*\*\*When to Consult a Lawyer:\*\*\s*\n', '\n|||SECTION:when_to_consult_lawyer|||\n'),
                (r'\n\s*When to Consult a Lawyer\s*:\s*\n', '\n|||SECTION:when_to_consult_lawyer|||\n'),
                (r'^\*\*When to Consult a Lawyer:\*\*\s*', '|||SECTION:when_to_consult_lawyer|||\n'),
                (r'^When to Consult a Lawyer\s*:\s*', '|||SECTION:when_to_consult_lawyer|||\n'),
                (r'\n\s*\*\*Disclaimers:\*\*\s*\n', '\n|||SECTION:disclaimers|||\n'),
                (r'\n\s*Disclaimers\s*:\s*\n', '\n|||SECTION:disclaimers|||\n'),
                (r'^\*\*Disclaimers:\*\*\s*', '|||SECTION:disclaimers|||\n'),
                (r'^Disclaimers\s*:\s*', '|||SECTION:disclaimers|||\n'),
            ]
        else:
            section_patterns = [
                (r'\n\s*\*\*وضاحت:\*\*\s*\n', '\n|||SECTION:explanation|||\n'),
                (r'\n\s*وضاحت\s*:\s*\n', '\n|||SECTION:explanation|||\n'),
                (r'^\*\*وضاحت:\*\*\s*', '|||SECTION:explanation|||\n'),
                (r'^وضاحت\s*:\s*', '|||SECTION:explanation|||\n'),
            ]
        
        # Try to split text by section markers first
        split_text = text
        for pattern, replacement in section_patterns:
            split_text = re.sub(pattern, replacement, split_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Split by common section markers
        lines = split_text.split('\n')
        current_section = None  # Start with None to capture initial content
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Check for section marker
            if '|||SECTION:' in line:
                section_match = re.search(r'\|\|\|SECTION:(\w+)\|\|\|', line)
                if section_match:
                    current_section = section_match.group(1)
                    # Remove the marker from the line
                    line = re.sub(r'\|\|\|SECTION:\w+\|\|\|', '', line).strip()
                    if not line:
                        continue
            
            if not line:
                # Preserve empty lines for formatting
                if current_section:
                    sections[current_section] += "\n"
                continue
            
            line_lower = line.lower()
            
            # Check for section headers (case-insensitive, flexible matching)
            is_header = False
            detected_section = None
            
            # Check explanation markers
            for marker in explanation_markers:
                if marker in line_lower:
                    detected_section = "explanation"
                    is_header = True
                    break
            
            # Check practical steps markers
            if not is_header:
                for marker in steps_markers:
                    if marker in line_lower:
                        detected_section = "practical_steps"
                        is_header = True
                        break
            
            # Check lawyer markers
            if not is_header:
                for marker in lawyer_markers:
                    if marker in line_lower:
                        detected_section = "when_to_consult_lawyer"
                        is_header = True
                        break
            
            # Check disclaimer markers
            if not is_header:
                for marker in disclaimer_markers:
                    if marker in line_lower:
                        detected_section = "disclaimers"
                        is_header = True
                        break
            
            # Check for markdown headers (standalone **text**)
            if not is_header and line.startswith("**") and line.endswith("**") and len(line) < 50:
                # Skip markdown headers
                is_header = True
                if current_section is None:
                    detected_section = "explanation"  # Default to explanation
            
            if is_header and detected_section:
                current_section = detected_section
                # Skip the header line itself (and clean it if it contains markdown)
                continue
            
            # If no section detected yet, default to explanation
            if current_section is None:
                current_section = "explanation"
            
            # Clean the line - remove any remaining markdown header markers
            clean_line = original_line.strip()
            
            # Remove section header text if it appears at the start of the line
            # Pattern: "Explanation" or "**Explanation:**" at start, followed by content
            header_patterns = [
                (r'^\*\*Explanation:\*\*\s*', ''),
                (r'^Explanation\s*:\s*', ''),
                (r'^Explanation\s+', ''),
                (r'^\*\*Practical Steps:\*\*\s*', ''),
                (r'^Practical Steps\s*:\s*', ''),
                (r'^Practical Steps\s+', ''),
                (r'^\*\*When to Consult a Lawyer:\*\*\s*', ''),
                (r'^When to Consult a Lawyer\s*:\s*', ''),
                (r'^When to Consult a Lawyer\s+', ''),
                (r'^\*\*Disclaimers:\*\*\s*', ''),
                (r'^Disclaimers\s*:\s*', ''),
                (r'^Disclaimers\s+', ''),
            ]
            
            for pattern, replacement in header_patterns:
                clean_line = re.sub(pattern, replacement, clean_line, flags=re.IGNORECASE)
            
            # Skip if line is empty after cleaning
            if not clean_line:
                continue
            
            # Add line to current section
            if sections[current_section]:
                sections[current_section] += "\n" + clean_line
            else:
                sections[current_section] = clean_line
        
        # Clean up sections - remove leading/trailing whitespace and duplicates
        for key in sections:
            content = sections[key].strip()
            
            # Remove duplicate lines (especially for lists)
            lines = content.split('\n')
            seen = set()
            cleaned_lines = []
            for line in lines:
                line_stripped = line.strip()
                # Create a hash for the line (normalize whitespace)
                line_hash = ' '.join(line_stripped.split())
                if line_hash and line_hash not in seen:
                    seen.add(line_hash)
                    cleaned_lines.append(line)
                elif not line_stripped:
                    # Preserve empty lines
                    cleaned_lines.append(line)
            
            # Remove duplicate numbered list items (1. 1. pattern)
            content = '\n'.join(cleaned_lines)
            # Fix patterns like "1. 1. text" -> "1. text"
            content = re.sub(r'^(\d+)\.\s+\1\.\s+', r'\1. ', content, flags=re.MULTILINE)
            
            sections[key] = content.strip()
        
        return sections
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        # This would typically query the database
        # For now, return empty list
        return []
    
    def save_conversation(self, session_id: str, question: str, response: Dict) -> bool:
        """Save conversation to database"""
        # This would typically save to database
        # For now, just log
        logger.info(f"Conversation saved for session {session_id}")
        return True 

    def _get_enhanced_faq_response(self, faq_result: Dict, original_question: str, is_roman_urdu: bool = True) -> Dict:
        """Use FAQ as primary source and enhance with AI"""
        try:
            # Parse the FAQ answer into sections
            faq_answer = faq_result['answer']
            faq_sections = self._parse_faq_answer(faq_answer, is_roman_urdu)
            
            # Build enhancement prompt using FAQ as base
            enhancement_prompt = self._build_enhancement_prompt(original_question, faq_result, is_roman_urdu)
            
            # Get AI enhancement
            ai_response = self.model.generate_content(enhancement_prompt)
            ai_sections = self._parse_legal_response(ai_response.text, original_question, is_roman_urdu)
            
            # Combine FAQ (primary) with AI enhancement
            combined_response = self._combine_faq_and_ai(faq_sections, ai_sections, is_roman_urdu)
            
            return {
                "success": True,
                "question": original_question,
                "response": combined_response,
                "metadata": {
                    "category": faq_result['category'],
                    "model": "FAQ_Enhanced_AI",
                    "language": "roman_urdu" if is_roman_urdu else "urdu",
                    "timestamp": str(Config.get_current_timestamp()),
                    "faq_used": True,
                    "response_type": "faq_enhanced",
                    "faq_id": faq_result.get('id'),
                    "usage_count": faq_result.get('usage_count', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error enhancing FAQ response: {str(e)}")
            # Fallback to direct FAQ response
            return self._get_direct_faq_response(faq_result, original_question, is_roman_urdu)
    
    def _build_enhancement_prompt(self, question: str, faq_result: Dict, is_roman_urdu: bool = True) -> str:
        """Build prompt to enhance FAQ with additional details"""
        
        if is_roman_urdu:
            prompt = f"""Aap Pakistan ke legal system ke expert hain. FAQ data already available hai, ise enhance karein.

Original Question: {question}

Available FAQ Data:
{faq_result['answer']}

Aap ko ye FAQ data ko enhance karna hai with additional details. IMPORTANT: Sirf Roman Urdu use karein, Urdu script nahi.

**Explanation:**
[FAQ ke basic points ko expand karein with more details - Roman Urdu mein]

**Practical Steps:**
[FAQ mein jo steps hain unko detail mein explain karein - Roman Urdu mein]

**When to Consult a Lawyer:**
[Lawyer consultation ke liye guidance - Roman Urdu mein]

**Important Legal Points:**
[FAQ mein jo important legal points hain unko highlight karein - Roman Urdu mein]

**Safety Tips:**
[Additional safety aur precaution tips - Roman Urdu mein]

CRITICAL RULES:
1. FAQ data ko primary source rakhein
2. FAQ ke basic points ko contradict na karein
3. Sirf additional information add karein
4. SIRF Roman Urdu use karein, Urdu script bilkul nahi
5. Practical aur actionable advice dein
6. FAQ ke exact numbers aur facts ko maintain karein (e.g., 90 din period)
7. Question aur answer format repeat na karein
8. Clean formatting use karein

Example: Agar FAQ mein "90 din ka period" hai to ise emphasize karein aur explain karein, lekin exact number change na karein."""
        else:
            prompt = f"""آپ پاکستان کے قانونی نظام کے ماہر ہیں۔ FAQ ڈیٹا پہلے سے موجود ہے، اسے بہتر بنائیں۔

اصل سوال: {question}

موجودہ FAQ ڈیٹا:
{faq_result['answer']}

آپ کو یہ FAQ ڈیٹا کو اضافی تفصیلات کے ساتھ بہتر بنانا ہے۔ اہم: صرف اردو استعمال کریں، رومن اردو نہیں۔

**وضاحت:**
[FAQ کے بنیادی نکات کو مزید تفصیلات کے ساتھ وسیع کریں - اردو میں]

**عملی اقدامات:**
[FAQ میں جو اقدامات ہیں انہیں تفصیل سے بیان کریں - اردو میں]

**وکیل سے مشورہ کب کریں:**
[وکیل سے مشورہ کے لیے رہنمائی - اردو میں]

**اہم قانونی نکات:**
[FAQ میں جو اہم قانونی نکات ہیں انہیں نمایاں کریں - اردو میں]

**حفاظتی نکات:**
[اضافی حفاظت اور احتیاطی تدابیر - اردو میں]

اہم قوانین:
1. FAQ ڈیٹا کو بنیادی ذریعہ رکھیں
2. FAQ کے بنیادی نکات کی مخالفت نہ کریں
3. صرف اضافی معلومات شامل کریں
4. صرف اردو استعمال کریں، رومن اردو بالکل نہیں
5. عملی اور قابل عمل مشورہ دیں
6. FAQ کے عین اعداد و حقائق کو برقرار رکھیں (مثلاً، 90 دن کا دورانیہ)
7. سوال اور جواب کا فارمیٹ دہرائیں نہیں
8. صاف فارمیٹ استعمال کریں"""

        return prompt
    
    def _combine_faq_and_ai(self, faq_sections: Dict, ai_sections: Dict, is_roman_urdu: bool = True) -> Dict:
        """Combine FAQ (primary) with AI enhancement"""
        
        # Start with FAQ as base
        combined = {
            "explanation": faq_sections.get("explanation", ""),
            "practical_steps": faq_sections.get("practical_steps", ""),
            "when_to_consult_lawyer": faq_sections.get("when_to_consult_lawyer", ""),
            "disclaimers": faq_sections.get("disclaimers", "")
        }
        
        # Clean up FAQ sections - remove any Q: A: formatting
        for key in combined:
            if combined[key]:
                # Remove Q: and A: prefixes if present
                combined[key] = self._clean_response_text(combined[key])
        
        # Add AI enhancement only if it doesn't contradict FAQ and maintains language consistency
        if ai_sections.get("explanation") and not self._contradicts_faq(combined["explanation"], ai_sections["explanation"]):
            # Check if AI response is in the same language
            if self._is_same_language(combined["explanation"], ai_sections["explanation"], is_roman_urdu):
                ai_explanation = self._clean_response_text(ai_sections["explanation"])
                if ai_explanation and ai_explanation != combined["explanation"]:
                    combined["explanation"] += "\n\n" + ai_explanation
        
        if ai_sections.get("practical_steps") and not self._contradicts_faq(combined["practical_steps"], ai_sections["practical_steps"]):
            if self._is_same_language(combined["practical_steps"], ai_sections["practical_steps"], is_roman_urdu):
                ai_steps = self._clean_response_text(ai_sections["practical_steps"])
                if ai_steps and ai_steps != combined["practical_steps"]:
                    combined["practical_steps"] += "\n\n" + ai_steps
        
        if ai_sections.get("when_to_consult_lawyer") and not self._contradicts_faq(combined["when_to_consult_lawyer"], ai_sections["when_to_consult_lawyer"]):
            if self._is_same_language(combined["when_to_consult_lawyer"], ai_sections["when_to_consult_lawyer"], is_roman_urdu):
                ai_lawyer = self._clean_response_text(ai_sections["when_to_consult_lawyer"])
                if ai_lawyer and ai_lawyer != combined["when_to_consult_lawyer"]:
                    combined["when_to_consult_lawyer"] += "\n\n" + ai_lawyer
        
        return combined
    
    def _contradicts_faq(self, faq_text: str, ai_text: str) -> bool:
        """Check if AI text contradicts FAQ text"""
        # Simple contradiction check - can be enhanced
        faq_lower = faq_text.lower()
        ai_lower = ai_text.lower()
        
        # Check for obvious contradictions
        contradictions = [
            ("90 din", "30 din"),
            ("notice", "no notice"),
            ("court", "no court"),
            ("lawyer", "no lawyer"),
            ("FIR", "no FIR")
        ]
        
        for faq_term, ai_term in contradictions:
            if faq_term in faq_lower and ai_term in ai_lower:
                return True
        
        return False 

    def _get_direct_faq_response(self, faq_result: Dict, original_question: str, is_roman_urdu: bool = True) -> Dict:
        """Return direct FAQ response without AI enhancement"""
        try:
            # Parse the FAQ answer into sections
            faq_answer = faq_result['answer']
            sections = self._parse_faq_answer(faq_answer, is_roman_urdu)
            
            return {
                "success": True,
                "question": original_question,
                "response": sections,
                "metadata": {
                    "category": faq_result['category'],
                    "model": "FAQ_Database",
                    "language": "roman_urdu" if is_roman_urdu else "urdu",
                    "timestamp": str(Config.get_current_timestamp()),
                    "faq_used": True,
                    "response_type": "direct_faq",
                    "faq_id": faq_result.get('id'),
                    "usage_count": faq_result.get('usage_count', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing FAQ response: {str(e)}")
            return {
                "success": True,
                "question": original_question,
                "response": {
                    "explanation": faq_result['answer'],
                    "practical_steps": "",
                    "when_to_consult_lawyer": "",
                    "disclaimers": "Ye information general guidance ke liye hai, legal advice nahi." if is_roman_urdu 
                                  else "یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔"
                },
                "metadata": {
                    "category": faq_result['category'],
                    "model": "FAQ_Database",
                    "language": "roman_urdu" if is_roman_urdu else "urdu",
                    "timestamp": str(Config.get_current_timestamp()),
                    "faq_used": True,
                    "response_type": "direct_faq"
                }
            }
    
    def _parse_faq_answer(self, faq_answer: str, is_roman_urdu: bool = True) -> Dict:
        """Parse FAQ answer into structured sections"""
        try:
            # Clean the FAQ answer first - remove any Q: A: formatting
            cleaned_answer = self._clean_response_text(faq_answer.strip())
            
            # If FAQ answer contains structured sections, extract them
            sections = self._extract_sections(cleaned_answer, is_roman_urdu)
            
            # If no structured sections found, create a proper structure
            if not any(sections.values()) or all(not section.strip() for section in sections.values()):
                # Split FAQ answer into logical parts
                parts = cleaned_answer.split('\n')
                
                # Find the main explanation (usually the first part)
                explanation = ""
                practical_steps = ""
                when_to_consult_lawyer = ""
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    # Skip if part contains question format
                    if part.startswith("Q:") or part.startswith("س:") or part.startswith("A:") or part.startswith("ج:"):
                        continue
                    
                    # If part contains numbered steps or practical guidance
                    if any(keyword in part.lower() for keyword in ['step', 'kar', 'karein', 'karna', '1.', '2.', '3.', '4.', '5.']):
                        if not practical_steps:
                            practical_steps = part
                        else:
                            practical_steps += "\n" + part
                    # If part contains lawyer-related keywords
                    elif any(keyword in part.lower() for keyword in ['lawyer', 'wakeel', 'consult', 'legal', 'court', 'adalt']):
                        if not when_to_consult_lawyer:
                            when_to_consult_lawyer = part
                        else:
                            when_to_consult_lawyer += "\n" + part
                    # Otherwise, treat as explanation
                    else:
                        if not explanation:
                            explanation = part
                        else:
                            explanation += "\n" + part
                
                # If no practical steps found, try to extract from explanation
                if not practical_steps and explanation:
                    # Look for numbered items in explanation
                    lines = explanation.split('\n')
                    explanation_lines = []
                    steps_lines = []
                    
                    for line in lines:
                        if any(line.strip().startswith(str(i) + '.') for i in range(1, 10)):
                            steps_lines.append(line.strip())
                        else:
                            explanation_lines.append(line.strip())
                    
                    if steps_lines:
                        practical_steps = '\n'.join(steps_lines)
                        explanation = '\n'.join(explanation_lines)
                
                sections = {
                    "explanation": explanation,
                    "practical_steps": practical_steps,
                    "when_to_consult_lawyer": when_to_consult_lawyer,
                    "disclaimers": ""
                }
            
            # Clean all sections
            for key in sections:
                if sections[key]:
                    sections[key] = self._clean_response_text(sections[key])
            
            return {
                "explanation": sections.get("explanation", cleaned_answer),
                "practical_steps": sections.get("practical_steps", ""),
                "when_to_consult_lawyer": sections.get("when_to_consult_lawyer", ""),
                "disclaimers": sections.get("disclaimers", 
                    "Ye information general guidance ke liye hai, legal advice nahi." if is_roman_urdu 
                    else "یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔")
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing FAQ answer: {str(e)}")
            return {
                "explanation": self._clean_response_text(faq_answer),
                "practical_steps": "",
                "when_to_consult_lawyer": "",
                "disclaimers": "Ye information general guidance ke liye hai, legal advice nahi." if is_roman_urdu 
                              else "یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔"
            } 

    def _is_same_language(self, text1: str, text2: str, expected_roman_urdu: bool = True) -> bool:
        """Check if both texts are in the same language format"""
        is_roman_1 = self._is_roman_urdu(text1)
        is_roman_2 = self._is_roman_urdu(text2)
        
        # Both should be in the same format as expected
        return (is_roman_1 == expected_roman_urdu) and (is_roman_2 == expected_roman_urdu) 

    def _clean_response_text(self, text: str) -> str:
        """Clean response text by removing Q: A: formatting and extra whitespace"""
        if not text:
            return text
        
        # Remove Q: and A: prefixes
        text = text.replace("Q:", "").replace("A:", "").replace("س:", "").replace("ج:", "")
        
        # Remove extra whitespace and newlines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("**") and not line.endswith("**"):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip() 