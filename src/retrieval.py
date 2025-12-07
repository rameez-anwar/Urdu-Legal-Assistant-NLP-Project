#!/usr/bin/env python3
"""
FAISS-based Retrieval Module for Roman Urdu Legal Assistant
Handles semantic search over legal document embeddings
"""

import os
import json
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentRetriever:
    """FAISS-based retriever for legal documents"""
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "flat"):
        """
        Initialize FAISS retriever
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("flat" or "ivf")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents = []  # Store document metadata
        # Use project root for paths
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.index_path = Path(project_root) / "data" / "embeddings" / "legal_index.faiss"
        self.metadata_path = Path(project_root) / "data" / "embeddings" / "legal_metadata.json"
        
        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_index(self, embedding_dim: Optional[int] = None):
        """Create FAISS index"""
        dim = embedding_dim or self.embedding_dim
        
        if self.index_type == "flat":
            # L2 distance index (exact search)
            self.index = faiss.IndexFlatL2(dim)
        elif self.index_type == "ivf":
            # Inverted file index (approximate, faster for large datasets)
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100)  # 100 clusters
            self.index.nprobe = 10  # Search in 10 clusters
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created FAISS index: {self.index_type}, dimension: {dim}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add documents to the index
        
        Args:
            embeddings: Numpy array of embeddings (n_docs, embedding_dim)
            documents: List of document metadata dictionaries
        """
        if self.index is None:
            self.create_index(embeddings.shape[1])
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to index
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            List of document dictionaries with similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty. Please add documents first.")
            return []
        
        # Ensure query is float32 and reshape
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Get results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                doc['distance'] = float(distance)
                doc['rank'] = i + 1
                results.append(doc)
        
        return results
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
                logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save metadata
            metadata = {
                'documents': self.documents,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'total_documents': len(self.documents)
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved metadata to {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                logger.warning("Index file not found. Creating new index.")
                return False
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.documents = metadata.get('documents', [])
                self.embedding_dim = metadata.get('embedding_dim', self.embedding_dim)
                self.index_type = metadata.get('index_type', self.index_type)
                
                logger.info(f"Loaded {len(self.documents)} documents from metadata")
                return True
            else:
                logger.warning("Metadata file not found.")
                return False
                
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def build_index_from_documents(self, documents: List[Dict], embedding_generator):
        """
        Build index from processed documents
        
        Args:
            documents: List of processed documents with segments
            embedding_generator: LegalEmbeddingGenerator instance
        """
        try:
            all_segments = []
            segment_metadata = []
            
            # Collect all segments
            for doc in documents:
                doc_meta = doc.get('metadata', {})
                for segment in doc.get('segments', []):
                    segment_text = segment.get('text', '')
                    if segment_text:
                        all_segments.append(segment_text)
                        segment_metadata.append({
                            'document_id': doc_meta.get('filename', 'unknown'),
                            'category': doc_meta.get('category', 'general'),
                            'segment_id': segment.get('id', 0),
                            'text': segment_text,
                            'length': segment.get('length', 0)
                        })
            
            if not all_segments:
                logger.warning("No segments found in documents")
                return
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(all_segments)} segments...")
            embeddings = embedding_generator.generate_embeddings_batch(all_segments)
            
            # Add to index
            self.add_documents(embeddings, segment_metadata)
            
            # Save index
            self.save_index()
            
            logger.info("✅ Index built and saved successfully")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        stats = {
            'total_documents': len(self.documents) if self.index else 0,
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type
        }
        return stats

if __name__ == "__main__":
    # Example usage
    retriever = LegalDocumentRetriever(embedding_dim=384)
    
    # Create index
    retriever.create_index()
    
    # Example: Add some documents (in real usage, use embeddings from embedding_generator)
    # embeddings = np.random.rand(10, 384).astype('float32')
    # documents = [{'text': f'Document {i}', 'id': i} for i in range(10)]
    # retriever.add_documents(embeddings, documents)
    
    print("Retrieval module ready!")

