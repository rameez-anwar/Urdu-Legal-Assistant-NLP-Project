# Legal Assistant - Roman Urdu Legal Assistant Chatbot

A comprehensive AI-powered legal assistant chatbot designed specifically for Pakistani legal system, supporting Roman Urdu language for better accessibility.

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based legal assistant that provides accurate, explainable legal guidance in Roman Urdu. The system combines:

- **Semantic Search**: FAISS-based document retrieval
- **AI Generation**: Google Gemini API for response generation
- **Knowledge Base**: Pakistani legal documents and FAQs
- **Multiple Interfaces**: Web UI, CLI, and REST API

## Key Features

- **Roman Urdu Support**: Full support for Roman Urdu input and output
- **RAG Architecture**: Retrieval-augmented generation for accurate, grounded responses
- **5 Legal Categories**: Family Law, Criminal Law, Property Law, Civil Law, Constitutional Law
- **FAQ Database**: Pre-built legal Q&A with semantic search
- **Multiple Interfaces**: Web chat, CLI, and REST API
- **Explainability**: Response transparency with source citations
- **Hallucination Reduction**: Grounded responses through document retrieval

## Requirements

- Python 3.8 or higher
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- 4-8 GB RAM (8 GB recommended)
- Optional: GPU for faster embeddings (CUDA compatible)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Legal Assistant Chatbot"
```

### 2. Create Virtual Environment

```bash
python -m venv urdu_legal_env
# Windows
urdu_legal_env\Scripts\activate
# Linux/Mac
source urdu_legal_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG=True
APP_NAME=Legal Assistant - Roman Urdu Legal Guide
DATABASE_URL=sqlite:///./urdu_legal.db
```

## Project Structure

```
Legal Assistant Chatbot/
├── chatbot_app.py           # Flask web app (main entry point)
├── app/                      # Application Layer
│   ├── api_server.py        # FastAPI REST API
│   ├── setup_data.py        # Data setup script
│   └── evaluate.py          # Evaluation script
├── src/                      # Core Modules
│   ├── config.py            # Configuration
│   ├── database.py          # Database operations
│   ├── preprocessing.py     # Data preprocessing
│   ├── embeddings.py        # Embedding generation
│   ├── retrieval.py         # FAISS retrieval
│   ├── rag_model.py         # RAG pipeline
│   └── gemini_service.py    # Gemini API service
├── data/                     # Data Directory
│   ├── legal_documents/     # Raw documents (PDF/TXT)
│   ├── processed/           # Processed JSON
│   └── embeddings/          # FAISS index
├── templates/                # Web Templates
│   └── chat.html            # Web UI
├── docs/                     # Documentation
├── requirements.txt          # Dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

See `PROJECT_STRUCTURE.md` for detailed structure documentation.

## Usage

### Option 1: Web Interface (Recommended)

```bash
python chatbot_app.py
```

Then open your browser and go to: `http://localhost:5000`

### Option 2: REST API

```bash
python app/api_server.py
```

API will be available at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

## Setting Up Legal Documents

### Step 1: Download Legal Documents

**Where to Download:**
- **Official Sources**: See `docs/DOCUMENT_SOURCES.md` for complete list
- **Key Sources**:
  - Law Commission of Pakistan: http://www.lawcommission.gov.pk/
  - National Assembly: http://www.na.gov.pk/
  - Supreme Court: http://www.supremecourt.gov.pk/

**Essential Documents:**
- Constitution of Pakistan (1973)
- Pakistan Penal Code (PPC)
- Code of Criminal Procedure (CrPC)
- Civil Procedure Code (CPC)
- Family Laws (Muslim Family Laws Ordinance)

### Step 2: Prepare Documents

Place your legal documents (PDFs or TXT files) in `data/legal_documents/`

**For Testing**: You can use sample text files (see `data/legal_documents/SAMPLE_DOCUMENTS.txt`)

### Step 3: Process Documents

Run the setup script:
```bash
python setup_data.py
```

This will:
- Extract text from PDFs
- Clean and segment documents
- Generate embeddings
- Build FAISS index

**Manual Processing** (if needed):
```python
from src.preprocessing import LegalDocumentPreprocessor
from src.rag_model import RAGLegalAssistant

preprocessor = LegalDocumentPreprocessor()
documents = preprocessor.process_directory("data/legal_documents", "pdf")

rag = RAGLegalAssistant()
rag.build_index_from_documents(documents)
```

**Note**: Run all commands from the project root directory.

## Configuration

Edit `config.py` to customize:

- **Legal Categories**: Add/modify legal categories
- **Sample Questions**: Update sample questions
- **Model Settings**: Change embedding or generation models
- **API Settings**: Configure ports and hosts

## API Endpoints

### REST API (FastAPI)

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/legal-guidance` - Get legal guidance
- `GET /api/categories` - Get legal categories
- `GET /api/conversation-history/{session_id}` - Get conversation history
- `POST /api/feedback` - Submit feedback
- `GET /api/faq/search` - Search FAQ database
- `GET /api/statistics` - Get system statistics

### Web API (Flask)

- `GET /` - Web chat interface
- `POST /api/chat` - Send chat message
- `GET /api/categories` - Get categories
- `GET /api/sample-questions` - Get sample questions
- `GET /api/faq/search` - Search FAQ

## Example Queries (Roman Urdu)

- "Mere shohar ne talaq ka notice bheja hai, main kya karun?"
- "Police mein FIR darj karane ka tarika kya hai?"
- "Jaidat ki taqseem ke liye kya karna chahiye?"
- "Mere pass warisat ka masla hai, main kya karun?"
- "Tenant ko kese nikalein?"

## Performance Metrics

Target Metrics (from Assignment 2):
- **F1-Score**: ≥ 85%
- **ROUGE-L**: ≥ 0.75
- **Response Time**: < 2 seconds
- **Hallucination Rate**: < 17%

### Evaluating Accuracy

To measure the chatbot's accuracy and performance:

```bash
# Install evaluation dependencies
pip install rouge-score nltk
python -m nltk.downloader punkt

# Run full evaluation
python app/evaluate.py

# Save results to file
python app/evaluate.py --output evaluation_results.json

# Evaluate specific aspects
python app/evaluate.py --faq-only      # FAQ matching accuracy
python app/evaluate.py --quality-only   # Response quality (ROUGE, BLEU)
python app/evaluate.py --retrieval-only # RAG retrieval accuracy
```

See `EVALUATION_GUIDE.md` for detailed instructions on creating test datasets and interpreting results.

## Development

### Adding New Legal Documents

1. Place PDF/TXT files in `data/legal_documents/`
2. Run preprocessing: `python preprocessing.py`
3. Rebuild index: Update embeddings and FAISS index

### Adding FAQ Entries

```python
from database import LegalDatabase

db = LegalDatabase()
db.save_faq(
    category="family_law",
    subcategory="Talaq aur Khula",
    question="Mere shohar ne talaq ka notice bheja hai, main kya karun?",
    answer="Agar aap ke shohar ne talaq ka notice bheja hai...",
    keywords="talaq, divorce, notice, shohar"
)
```

## Troubleshooting

### Issue: "GEMINI_API_KEY not found"
- Solution: Create `.env` file with your API key

### Issue: "FAISS index not found"
- Solution: Build index first using `rag.build_index_from_documents()`

### Issue: "Model download failed"
- Solution: Check internet connection, try manual download from Hugging Face

### Issue: "Out of memory"
- Solution: Use smaller batch size in embeddings or use CPU-only FAISS

## License

This project is for educational purposes as part of NLP Lab coursework.

## Authors

- Rameez Anwar

## Acknowledgments

- Google Gemini API for AI generation
- Hugging Face for transformer models
- FAISS for efficient similarity search
- Pakistani legal system documentation

---

**Disclaimer**: This system provides general legal guidance only, not legal advice. Users should consult qualified lawyers for complex legal matters.

