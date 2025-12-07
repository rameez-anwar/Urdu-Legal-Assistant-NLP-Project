# Legal Documents Directory

Place your legal documents (PDF or TXT files) in this directory.

## Supported Formats
- PDF files (.pdf)
- Text files (.txt)

## Document Processing

After adding documents, run:
```bash
python setup_data.py
```

This will:
1. Process all documents
2. Generate embeddings
3. Build FAISS index for semantic search

## Example Documents
- Constitution of Pakistan
- Pakistan Penal Code (PPC)
- Civil Procedure Code (CPC)
- Family Law documents
- Property Law documents
- Any other Pakistani legal documents

## Notes
- Documents should be in Roman Urdu or English
- Large documents will be automatically segmented
- Processing may take time depending on document size

