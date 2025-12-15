import os
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalEmbeddingGenerator:
    """Generate embeddings for legal documents using transformer models"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            # Use sentence-transformers for easier usage
            self.model = SentenceTransformer(model_name, device=self.device)
            self.use_sentence_transformers = True
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer, trying transformers: {str(e)}")
            try:
                # Fallback to transformers library
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.use_sentence_transformers = False
                logger.info(f"Loaded transformers model: {model_name}")
            except Exception as e2:
                logger.error(f"Failed to load model: {str(e2)}")
                raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            if self.use_sentence_transformers:
                embedding = self.model.encode(text, convert_to_numpy=True)
            else:
                # Use transformers library
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                       max_length=512, padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(384) if 'MiniLM' in self.model_name else np.zeros(768)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts in batches"""
        try:
            if self.use_sentence_transformers:
                embeddings = self.model.encode(
                    texts, 
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
            else:
                # Process in batches
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = []
                    
                    for text in batch_texts:
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                               max_length=512, padding=True).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                            batch_embeddings.append(embedding)
                    
                    embeddings.extend(batch_embeddings)
                
                embeddings = np.array(embeddings)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return np.array([])
    
    def generate_document_embeddings(self, documents: List[Dict]) -> List[Dict]:
        """Generate embeddings for document segments"""
        try:
            all_segments = []
            segment_to_doc = []
            
            # Collect all segments
            for doc_idx, doc in enumerate(documents):
                for segment in doc.get('segments', []):
                    all_segments.append(segment['text'])
                    segment_to_doc.append((doc_idx, segment['id']))
            
            if not all_segments:
                logger.warning("No segments found in documents")
                return documents
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(all_segments)} segments...")
            embeddings = self.generate_embeddings_batch(all_segments)
            
            # Add embeddings back to documents
            for idx, (doc_idx, seg_id) in enumerate(segment_to_doc):
                if idx < len(embeddings):
                    # Find the segment in the document
                    for segment in documents[doc_idx]['segments']:
                        if segment['id'] == seg_id:
                            segment['embedding'] = embeddings[idx].tolist()
                            break
            
            logger.info("Embeddings added to documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error generating document embeddings: {str(e)}")
            return documents
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.use_sentence_transformers:
            return self.model.get_sentence_embedding_dimension()
        else:
            # Default BERT dimension
            return 768
    
    def save_model(self, save_path: str):
        """Save the model to disk"""
        try:
            if self.use_sentence_transformers:
                self.model.save(save_path)
            else:
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    # Example usage
    generator = LegalEmbeddingGenerator()
    
    # Test single embedding
    text = "Mere shohar ne talaq ka notice bheja hai, main kya karun?"
    embedding = generator.generate_embedding(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    
    # Test batch embeddings
    texts = [
        "Talaq ka process kya hai?",
        "FIR darj karane ka tarika?",
        "Jaidat ki taqseem ke liye kya karna chahiye?"
    ]
    embeddings = generator.generate_embeddings_batch(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")

