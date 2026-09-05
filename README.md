# Document Q&A RAG API

A production-ready **Document Q&A RAG Service** built with **FastAPI**, **LangChain**, **FAISS**, and **Google Gemini API** (with automatic offline fallback). It allows users to upload documents (`.pdf` or `.txt`), chunk and vector-index them into a persisted FAISS store, and perform grounded natural-language question answering with source citations.

---

## 🏗️ Architecture Diagram

```
+-----------------------------------------------------------------------+
|                            User Interface                             |
|         Single-Page App (Vanilla HTML/CSS/JS) or HTTP Client           |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                            FastAPI Server                             |
|  POST /documents/upload | GET /documents | DELETE /documents/{id}     |
|  POST /query            | GET /health    | GET / (Static Files)       |
+-----------------+---------------------------------+-------------------+
                  |                                 |
                  v                                 v
+-----------------+---------------+   +-------------+-------------------+
|      Document Ingestion Engine  |   |    RAG Retrieval & QA Engine  |
|  • PyPDF / Text Loader          |   |  • Top-k Vector Similarity Search |
|  • RecursiveCharacterTextSplitter|  |  • Strict Grounded Prompting    |
|  • Metadata Tracking (File/Page)|   |  • Gemini LLM / Offline Engine  |
+-----------------+---------------+   +-------------+-------------------+
                  |                                 |
                  v                                 v
+-----------------+---------------------------------+-------------------+
|                         Persistence Layer                             |
|  • Uploaded Files: app/storage/                                       |
|  • Vector Store: faiss_index/ (index.faiss, index.pkl)                |
|  • Metadata Store: app/storage/documents.json                         |
+-----------------------------------------------------------------------+
```

---

## ⚡ Tech Stack

- **Framework:** FastAPI, Uvicorn
- **Orchestration:** LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`)
- **Vector Database:** FAISS (`faiss-cpu`)
- **LLM & Embeddings:** Google Gemini API (`langchain-google-genai`), with automatic offline fallback engine
- **Document Parsers:** `pypdf` (PDF), UTF-8 text parser (`.txt`)
- **Frontend:** Plain HTML/CSS/JS single-page UI (served as static files from FastAPI)
- **Testing:** `pytest`, `httpx`

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- (Optional) Google Gemini API Key from Google AI Studio.

### 2. Environment Setup
Clone the repository and install dependencies:

```bash
# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Edit `.env` and insert your Gemini API Key (leave empty to run in offline fallback mode):

```env
GOOGLE_API_KEY=your_gemini_api_key_here
STORAGE_DIR=app/storage
FAISS_INDEX_DIR=faiss_index
CHUNK_SIZE=800
CHUNK_OVERLAP=100
DEFAULT_TOP_K=4
```

### 3. Run the Service

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

- **Interactive Web Demo UI:** Open `http://localhost:8000/` in your browser.
- **Interactive Swagger Docs:** Open `http://localhost:8000/docs` in your browser.

---

## 🧪 Running Automated Tests

Run the comprehensive pytest test suite covering upload, query, document listing, vector deletion, and error handling:

```bash
python -m pytest -v
```

---

## 📡 API Reference & `curl` Examples

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Response:**
```json
{
  "status": "ok",
  "document_count": 1,
  "total_chunks": 4,
  "mode": "gemini"
}
```

### 2. Upload and Ingest Document
Upload a `.pdf` or `.txt` file for chunking and vector indexing:

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@sample.pdf"
```
**Response:**
```json
{
  "document_id": "doc_a1b2c3d4e5f6",
  "filename": "sample.pdf",
  "chunk_count": 4,
  "message": "Successfully ingested 'sample.pdf' with 4 chunks."
}
```

### 3. List Ingested Documents
```bash
curl -X GET "http://localhost:8000/documents"
```
**Response:**
```json
{
  "documents": [
    {
      "document_id": "doc_a1b2c3d4e5f6",
      "filename": "sample.pdf",
      "upload_timestamp": "2026-09-05T22:00:00+00:00",
      "chunk_count": 4
    }
  ]
}
```

### 4. Query Documents (Natural Language Q&A)
Ask a natural language question with optional document ID filtering:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the primary function of FAISS?",
    "document_id": "doc_a1b2c3d4e5f6"
  }'
```
**Response:**
```json
{
  "answer": "FAISS is a library developed for efficient similarity search and clustering of dense vectors.",
  "sources": [
    {
      "filename": "sample.pdf",
      "page": 1,
      "snippet": "FAISS is a library developed for efficient similarity search..."
    }
  ]
}
```

### 5. Delete Document
Remove a document, its physical file, and its vectors from the FAISS index:

```bash
curl -X DELETE "http://localhost:8000/documents/doc_a1b2c3d4e5f6"
```
**Response:**
```json
{
  "message": "Successfully deleted document 'sample.pdf' (4 chunks removed).",
  "document_id": "doc_a1b2c3d4e5f6"
}
```

---

## 🎨 Single-Page Frontend Demo

The backend serves a lightweight demo interface directly at `http://localhost:8000/`:
- **Left Panel (Documents):** File upload widget, real-time ingested documents list, and inline document deletion.
- **Right Panel (Ask):** Interactive Q&A chat feed with document context selection, real-time loading state ("Thinking…"), grounded answer display, and muted source snippet citations.

---

## 📂 Project Structure

```
rag-doc-qa/
├── app/
│   ├── main.py              # FastAPI app setup, route definitions, static mounting
│   ├── config.py            # Environment configuration & path initialization
│   ├── ingestion.py         # File parsing (PyPDF/TXT), RecursiveTextSplitter, chunking
│   ├── retrieval.py         # Vector similarity search & strict grounded QA prompt chain
│   ├── vector_store.py      # FAISS index persistence, metadata management, fallback embeddings
│   ├── models.py            # Pydantic API request & response schemas
│   ├── storage/             # Local file storage (gitignored)
│   └── static/              # Single-page demo UI (index.html, style.css, app.js)
├── faiss_index/             # Persisted FAISS vector index files (gitignored)
├── tests/
│   └── test_api.py          # Pytest integration tests for all API endpoints
├── requirements.txt         # Project dependencies
├── .env.example             # Template environment file
├── README.md                # Project documentation
└── .gitignore               # Version control ignore rules
```
