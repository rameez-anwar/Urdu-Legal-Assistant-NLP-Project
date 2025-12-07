#!/usr/bin/env python3
"""
Data Preprocessing Module for Roman Urdu Legal Assistant
Handles text extraction, cleaning, tokenization, and augmentation
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2
import pdfminer
from pdfminer.high_level import extract_text as pdf_extract_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentPreprocessor:
    """Preprocess legal documents for the chatbot"""
    
    def __init__(self, output_dir: str = None):
        """Initialize preprocessor"""
        if output_dir is None:
            # Default to project root/data/processed
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, "data", "processed")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            # Try pdfminer first (better for complex PDFs)
            text = pdf_extract_text(pdf_path)
            if text.strip():
                return text
            
            # Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean legal text by removing unnecessary elements"""
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*/\s*\d+', '', text)
        
        # Remove citation patterns (e.g., [1], (2023), etc.) but keep content
        text = re.sub(r'\[(\d+)\]', r'\1', text)  # Keep number, remove brackets
        text = re.sub(r'\([0-9]{4}\)', '', text)
        
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
        
        # Preserve all text including Roman Urdu, Urdu script, and English
        # Don't remove special characters that might be part of Roman Urdu
        
        return text.strip()
    
    def segment_document(self, text: str, max_length: int = 1000) -> List[Dict]:
        """Segment document into meaningful chunks"""
        segments = []
        
        # First, try to split by multiple newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n+', text)
        
        # If no paragraphs found, split by single newlines
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
        
        current_segment = ""
        segment_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 10:  # Skip very short paragraphs
                continue
            
            # If current segment + new paragraph exceeds max_length, save current segment
            combined_length = len(current_segment) + len(para) + 2  # +2 for newline
            if combined_length > max_length and current_segment:
                segments.append({
                    'id': segment_id,
                    'text': current_segment.strip(),
                    'length': len(current_segment)
                })
                segment_id += 1
                current_segment = para
            # If single paragraph is too long, split it by sentences
            elif len(para) > max_length:
                # Save current segment first if exists
                if current_segment:
                    segments.append({
                        'id': segment_id,
                        'text': current_segment.strip(),
                        'length': len(current_segment)
                    })
                    segment_id += 1
                    current_segment = ""
                
                # Split long paragraph by sentences
                sentences = re.split(r'[.!?]\s+', para)
                temp_segment = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(temp_segment) + len(sentence) + 2 > max_length and temp_segment:
                        segments.append({
                            'id': segment_id,
                            'text': temp_segment.strip(),
                            'length': len(temp_segment)
                        })
                        segment_id += 1
                        temp_segment = sentence
                    else:
                        if temp_segment:
                            temp_segment += ". " + sentence
                        else:
                            temp_segment = sentence
                
                if temp_segment:
                    current_segment = temp_segment
            else:
                # Add paragraph to current segment
                if current_segment:
                    current_segment += "\n\n" + para
                else:
                    current_segment = para
        
        # Add remaining segment
        if current_segment and current_segment.strip():
            segments.append({
                'id': segment_id,
                'text': current_segment.strip(),
                'length': len(current_segment)
            })
        
        # Ensure at least one segment
        if not segments and text.strip():
            segments.append({
                'id': 0,
                'text': text.strip(),
                'length': len(text)
            })
        
        return segments
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Roman Urdu, Urdu script, or English"""
        text_lower = text.lower()
        
        # Roman Urdu keywords
        roman_urdu_keywords = ['talaq', 'shohar', 'jaidat', 'warisat', 'qanoon', 
                              'adalat', 'wakeel', 'mehr', 'nikah', 'khula', 
                              'fir', 'police', 'court', 'lawyer', 'notice']
        
        # Urdu script keywords (Unicode range)
        urdu_script_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        
        # Count Roman Urdu keywords
        roman_urdu_count = sum(1 for keyword in roman_urdu_keywords if keyword in text_lower)
        
        # Determine language
        if urdu_script_count > len(text) * 0.1:  # More than 10% Urdu script
            return 'urdu'
        elif roman_urdu_count >= 3:  # At least 3 Roman Urdu keywords
            return 'roman_urdu'
        else:
            return 'english'
    
    def extract_metadata(self, text: str, filename: str) -> Dict:
        """Extract metadata from document"""
        # Detect language
        language = self.detect_language(text)
        
        metadata = {
            'filename': filename,
            'total_length': len(text),
            'word_count': len(text.split()),
            'language': language
        }
        
        # Improved category detection
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Family Law keywords
        family_keywords = ['family', 'marriage', 'talaq', 'divorce', 'nikah', 'khula', 
                          'mehr', 'dower', 'custody', 'khandani', 'shohar', 'biwi',
                          'muslim family laws', 'ordinance 1961']
        
        # Criminal Law keywords
        criminal_keywords = ['penal', 'criminal', 'fir', 'police', 'bail', 'warrant',
                            'jurm', 'arrest', 'punishment', 'offence', 'crime']
        
        # Property Law keywords
        property_keywords = ['property', 'jaidat', 'land', 'real estate', 'transfer',
                            'lease', 'rent', 'tenant', 'eviction', 'partition']
        
        # Civil Law keywords
        civil_keywords = ['civil', 'contract', 'agreement', 'breach', 'damages',
                         'recovery', 'madani', 'suit', 'plaintiff', 'defendant']
        
        # Constitutional Law keywords
        constitutional_keywords = ['constitution', 'fundamental rights', 'dastoor',
                                  'government', 'election', 'discrimination']
        
        # Check category based on keywords
        family_score = sum(1 for kw in family_keywords if kw in text_lower or kw in filename_lower)
        criminal_score = sum(1 for kw in criminal_keywords if kw in text_lower or kw in filename_lower)
        property_score = sum(1 for kw in property_keywords if kw in text_lower or kw in filename_lower)
        civil_score = sum(1 for kw in civil_keywords if kw in text_lower or kw in filename_lower)
        constitutional_score = sum(1 for kw in constitutional_keywords if kw in text_lower or kw in filename_lower)
        
        # Determine category based on highest score
        scores = {
            'family_law': family_score,
            'criminal_law': criminal_score,
            'property_law': property_score,
            'civil_law': civil_score,
            'constitutional_law': constitutional_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            metadata['category'] = max(scores, key=scores.get)
        else:
            metadata['category'] = 'general'
        
        return metadata
    
    def process_document(self, file_path: str, file_type: str = 'pdf') -> Optional[Dict]:
        """Process a single document"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Extract text
            if file_type.lower() == 'pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_type.lower() == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return None
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return None
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Extract metadata
            filename = Path(file_path).stem
            metadata = self.extract_metadata(cleaned_text, filename)
            
            # Segment document
            segments = self.segment_document(cleaned_text)
            
            # Create document structure
            document = {
                'metadata': metadata,
                'segments': segments,
                'total_segments': len(segments)
            }
            
            # Save processed document
            output_file = self.output_dir / f"{filename}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Processed document saved to {output_file}")
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return None
    
    def process_directory(self, directory_path: str, file_type: str = 'pdf') -> List[Dict]:
        """Process all documents in a directory"""
        directory = Path(directory_path)
        processed_documents = []
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return processed_documents
        
        # Find all files of specified type
        if file_type.lower() == 'pdf':
            files = list(directory.glob('*.pdf'))
        elif file_type.lower() == 'txt':
            files = list(directory.glob('*.txt'))
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return processed_documents
        
        logger.info(f"Found {len(files)} files to process")
        
        for file_path in files:
            document = self.process_document(str(file_path), file_type)
            if document:
                processed_documents.append(document)
        
        logger.info(f"Successfully processed {len(processed_documents)} documents")
        return processed_documents
    
    def augment_text(self, text: str, method: str = 'synonym') -> str:
        """Augment text using various methods"""
        # Simple synonym replacement for Roman Urdu legal terms
        synonyms = {
            'talaq': ['divorce', 'separation'],
            'shohar': ['husband', 'spouse'],
            'jaidat': ['property', 'real estate'],
            'warisat': ['inheritance', 'legacy'],
            'qanoon': ['law', 'legal'],
            'adalat': ['court', 'tribunal'],
            'wakeel': ['lawyer', 'attorney'],
            'police': ['police', 'law enforcement']
        }
        
        if method == 'synonym':
            words = text.split()
            augmented = []
            for word in words:
                word_lower = word.lower()
                if word_lower in synonyms:
                    # Randomly choose synonym (for now, use first)
                    augmented.append(synonyms[word_lower][0])
                else:
                    augmented.append(word)
            return ' '.join(augmented)
        
        return text

if __name__ == "__main__":
    # Example usage
    preprocessor = LegalDocumentPreprocessor()
    
    # Process a single document
    # document = preprocessor.process_document("path/to/document.pdf", "pdf")
    
    # Process a directory
    # documents = preprocessor.process_directory("data/legal_documents", "pdf")
    
    print("Preprocessing module ready!")

