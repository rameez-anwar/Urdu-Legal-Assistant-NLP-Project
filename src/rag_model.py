import os
import logging
from typing import List, Dict, Optional
from src.config import Config
from src.embeddings import LegalEmbeddingGenerator
from src.retrieval import LegalDocumentRetriever
from src.gemini_service import GeminiLegalService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGLegalAssistant:
    """RAG-based legal assistant combining retrieval and generation"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 top_k: int = 5):
        """
        Initialize RAG assistant
        
        Args:
            embedding_model: Model name for embeddings
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        
        # Initialize components
        logger.info("Initializing RAG components...")
        self.embedding_generator = LegalEmbeddingGenerator(model_name=embedding_model)
        self.retriever = LegalDocumentRetriever(
            embedding_dim=self.embedding_generator.get_embedding_dimension()
        )
        self.gemini_service = GeminiLegalService()
        
        # Load existing index if available
        if not self.retriever.load_index():
            logger.warning("No existing index found. Please build index first.")
        
        logger.info("RAG assistant initialized")
    
    def retrieve_relevant_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search in index
            results = self.retriever.search(query_embedding, k=self.top_k)
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def build_context(self, retrieved_docs: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get('text', '')
            category = doc.get('category', 'general')
            similarity = doc.get('similarity_score', 0)
            
            context_parts.append(
                f"[Document {i} - {category} - Similarity: {similarity:.2f}]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, retrieved_docs: List[Dict] = None) -> Dict:
        """
        Generate response using RAG
        
        Args:
            query: User query in Roman Urdu
            retrieved_docs: Optional pre-retrieved documents
        
        Returns:
            Response dictionary with explanation, steps, etc.
        """
        try:
            # Retrieve documents if not provided
            if retrieved_docs is None:
                retrieved_docs = self.retrieve_relevant_documents(query)
            
            # Build context from retrieved documents
            context = self.build_context(retrieved_docs)
            
            # Get category from query or retrieved docs
            category = None
            if retrieved_docs:
                # Use most relevant document's category
                category = retrieved_docs[0].get('category', None)
            
            # Enhance Gemini prompt with retrieved context
            enhanced_response = self._get_rag_enhanced_response(
                query, context, category, retrieved_docs
            )
            
            # Add metadata
            enhanced_response['metadata']['retrieval_used'] = True
            enhanced_response['metadata']['retrieved_docs_count'] = len(retrieved_docs)
            enhanced_response['metadata']['retrieved_docs'] = [
                {
                    'category': doc.get('category'),
                    'similarity': doc.get('similarity_score'),
                    'text_preview': doc.get('text', '')[:100] + '...'
                }
                for doc in retrieved_docs[:3]  # Top 3 for metadata
            ]
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            # Fallback to direct Gemini response
            return self.gemini_service.get_legal_guidance(query)
    
    def _get_rag_enhanced_response(self, query: str, context: str, category: str = None, 
                                   retrieved_docs: List[Dict] = None) -> Dict:
        """Get response enhanced with retrieved context"""
        try:
            # Build enhanced prompt
            is_roman_urdu = self.gemini_service._is_roman_urdu(query)
            
            if is_roman_urdu:
                prompt = f"""Aap Pakistan ke legal system ke expert hain. Aap ko Roman Urdu mein jawab dena hai.

User ka sawal: {query}

Relevant Legal Documents (Retrieved from database):
{context}

Aap ko ye format mein jawab dena hai:

**Explanation:**
[Roman Urdu mein detailed explanation - retrieved documents ke basis par]

**Practical Steps:**
[Roman Urdu mein step-by-step guide - retrieved documents se practical steps]

**When to Consult a Lawyer:**
[Roman Urdu mein guidance - complex cases ke liye]

**Relevant Legal References:**
[Retrieved documents se relevant legal points - Roman Urdu mein]

**Disclaimers:**
Ye information general guidance ke liye hai, legal advice nahi. Complex cases mein lawyer se consult karein.

IMPORTANT INSTRUCTIONS:
1. Sirf Roman Urdu use karein (Urdu script nahi)
2. Retrieved documents ke information ko use karein
3. Agar retrieved documents mein relevant information nahi hai to general guidance dein
4. Hallucination avoid karein - sirf retrieved context aur general knowledge use karein
5. Simple aur samajhne mein aasan language use karein
6. Practical aur actionable advice dein
7. Legal jargon avoid karein
8. Safety aur legal rights emphasize karein

Please provide a helpful, practical response in Roman Urdu based on the retrieved legal documents."""
            else:
                # Urdu script prompt (similar structure)
                prompt = f"""آپ پاکستان کے قانونی نظام کے ماہر ہیں۔ آپ کو اردو میں جواب دینا ہے۔

صارف کا سوال: {query}

متعلقہ قانونی دستاویزات (ڈیٹا بیس سے حاصل کردہ):
{context}

آپ کو یہ فارمیٹ میں جواب دینا ہے:

**وضاحت:**
[اردو میں تفصیلی وضاحت - حاصل کردہ دستاویزات کی بنیاد پر]

**عملی اقدامات:**
[اردو میں مرحلہ وار رہنمائی - حاصل کردہ دستاویزات سے عملی اقدامات]

**وکیل سے مشورہ کب کریں:**
[اردو میں رہنمائی - پیچیدہ معاملات کے لیے]

**متعلقہ قانونی حوالہ جات:**
[حاصل کردہ دستاویزات سے متعلقہ قانونی نکات - اردو میں]

**تنبیہ:**
یہ معلومات عام رہنمائی کے لیے ہے، قانونی مشورہ نہیں۔ پیچیدہ معاملات میں وکیل سے مشورہ کریں۔

اہم ہدایات:
1. صرف اردو استعمال کریں
2. حاصل کردہ دستاویزات کی معلومات استعمال کریں
3. اگر حاصل کردہ دستاویزات میں متعلقہ معلومات نہیں ہے تو عام رہنمائی دیں
4. Hallucination سے بچیں - صرف حاصل کردہ سیاق و سباق اور عام علم استعمال کریں
5. سادہ اور سمجھنے میں آسان زبان استعمال کریں
6. عملی اور قابل عمل مشورہ دیں
7. قانونی اصطلاحات سے بچیں
8. حفاظت اور قانونی حقوق پر زور دیں

براہ کرم حاصل کردہ قانونی دستاویزات کی بنیاد پر اردو میں مددگار اور عملی جواب دیں۔"""
            
            # Get response from Gemini
            try:
                response = self.gemini_service.model.generate_content(prompt)
                parsed_response = self.gemini_service._parse_legal_response(
                    response.text, query, is_roman_urdu
                )
            except Exception as e:
                logger.error(f"Error generating Gemini response: {str(e)}")
                # Fallback to direct service call
                return self.gemini_service.get_legal_guidance(query, category)
            
            return {
                "success": True,
                "question": query,
                "response": parsed_response,
                "metadata": {
                    "category": category,
                    "model": "RAG_" + Config.GEMINI_MODEL,
                    "language": "roman_urdu" if is_roman_urdu else "urdu",
                    "timestamp": str(Config.get_current_timestamp()),
                    "retrieval_used": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG enhanced response: {str(e)}")
            # Fallback
            return self.gemini_service.get_legal_guidance(query)
    
    def build_index_from_documents(self, documents: List[Dict]):
        """Build FAISS index from documents"""
        try:
            logger.info("Building FAISS index from documents...")
            self.retriever.build_index_from_documents(documents, self.embedding_generator)
            logger.info("Index built successfully")
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")

if __name__ == "__main__":
    # Example usage
    rag = RAGLegalAssistant()
    
    # Test query
    query = "Mere shohar ne talaq ka notice bheja hai, main kya karun?"
    response = rag.generate_response(query)
    
    print("RAG response generated!")
    print(f"Retrieved docs: {response['metadata'].get('retrieved_docs_count', 0)}")

