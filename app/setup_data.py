#!/usr/bin/env python3
"""
Setup script to initialize the legal document database
Processes documents and builds FAISS index for RAG
"""

import os
import sys
from pathlib import Path
import sys
import os

# Add parent directory and src to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.preprocessing import LegalDocumentPreprocessor
from src.rag_model import RAGLegalAssistant

def setup_data():
    """Setup data processing and index building"""
    print("🔧 Setting up Legal Assistant Legal Document Database...")
    print("="*60)
    
    # Check if data directory exists (relative to project root)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "legal_documents"
    if not data_dir.exists():
        print(f"📁 Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("⚠️  No documents found. Please add PDF/TXT files to data/legal_documents/")
        print("   Then run this script again.")
        return
    
    # Find documents
    pdf_files = list(data_dir.glob("*.pdf"))
    txt_files = list(data_dir.glob("*.txt"))
    total_files = len(pdf_files) + len(txt_files)
    
    if total_files == 0:
        print("⚠️  No documents found in data/legal_documents/")
        print("   Please add PDF or TXT files and run again.")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files and {len(txt_files)} TXT files")
    
    # Initialize preprocessor (output to project root data/processed)
    preprocessor = LegalDocumentPreprocessor(
        output_dir=str(project_root / "data" / "processed")
    )
    
    # Process PDF files
    processed_docs = []
    if pdf_files:
        print("\n📄 Processing PDF files...")
        pdf_docs = preprocessor.process_directory(str(data_dir), "pdf")
        processed_docs.extend(pdf_docs)
    
    # Process TXT files
    if txt_files:
        print("\n📄 Processing TXT files...")
        txt_docs = preprocessor.process_directory(str(data_dir), "txt")
        processed_docs.extend(txt_docs)
    
    if not processed_docs:
        print("❌ No documents were successfully processed")
        return
    
    print(f"\n✅ Successfully processed {len(processed_docs)} documents")
    
    # Build FAISS index
    print("\n🔍 Building FAISS index for semantic search...")
    try:
        rag = RAGLegalAssistant()
        rag.build_index_from_documents(processed_docs)
        print("✅ FAISS index built successfully!")
        print(f"📊 Index statistics: {rag.retriever.get_statistics()}")
    except Exception as e:
        print(f"❌ Error building index: {str(e)}")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")
        return
    
    print("\n" + "="*60)
    print("✅ Setup completed successfully!")
    print("🚀 You can now run the chatbot:")
    print("   - Web UI: python chatbot_app.py")
    print("   - CLI: python main.py")
    print("   - API: python api_server.py")

if __name__ == "__main__":
    setup_data()

