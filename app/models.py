from typing import List, Optional
from pydantic import BaseModel, Field

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    message: str

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    upload_timestamp: str
    chunk_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

class DeleteDocumentResponse(BaseModel):
    message: str
    document_id: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The natural language question to ask.")
    document_id: Optional[str] = Field(None, description="Optional document ID to restrict query context.")

class SourceChunk(BaseModel):
    filename: str
    page: int
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class HealthResponse(BaseModel):
    status: str
    document_count: int
    total_chunks: int
    mode: str
