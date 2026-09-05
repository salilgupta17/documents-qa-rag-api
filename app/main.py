from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Path as APIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentInfo,
    DeleteDocumentResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse
)
from app.ingestion import process_and_ingest_file
from app.retrieval import query_rag_pipeline
from app.vector_store import (
    vector_store_manager,
    get_all_documents,
    remove_document_metadata
)

app = FastAPI(
    title="Document Q&A RAG API",
    description="FastAPI service for document ingestion, vector retrieval with FAISS, and grounded QA using LangChain and Gemini.",
    version="1.0.0"
)

# Enable CORS for local testing/demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    docs = get_all_documents()
    total_chunks = vector_store_manager.get_total_chunks()
    return HealthResponse(
        status="ok",
        document_count=len(docs),
        total_chunks=total_chunks,
        mode=vector_store_manager.mode
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile = File(...)):
    result = process_and_ingest_file(file)
    return DocumentUploadResponse(**result)


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    docs = get_all_documents()
    doc_infos = [
        DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            upload_timestamp=d["upload_timestamp"],
            chunk_count=d["chunk_count"]
        ) for d in docs
    ]
    return DocumentListResponse(documents=doc_infos)


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str = APIPath(..., description="The ID of the document to delete")):
    doc_info = remove_document_metadata(document_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found.")

    # Delete physical file
    file_path = Path(doc_info.get("file_path", ""))
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete physical file {file_path}: {e}")

    # Delete vectors from FAISS index
    deleted_chunks = vector_store_manager.delete_document_vectors(document_id)

    return DeleteDocumentResponse(
        message=f"Successfully deleted document '{doc_info.get('filename')}' ({deleted_chunks} chunks removed).",
        document_id=document_id
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    result = query_rag_pipeline(
        question=request.question,
        document_id=request.document_id
    )
    return QueryResponse(**result)


# Mount static files for single-page frontend demo
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
