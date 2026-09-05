import datetime
import uuid
from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

from app.config import STORAGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.vector_store import vector_store_manager, add_document_metadata

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def process_and_ingest_file(file: UploadFile) -> dict:
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .pdf and .txt files are supported."
        )

    # Generate document ID and destination path
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    file_path = STORAGE_DIR / f"{doc_id}_{filename}"

    try:
        content_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    if not content_bytes or len(content_bytes.strip()) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save to local storage
    with open(file_path, "wb") as f:
        f.write(content_bytes)

    documents: List[Document] = []

    if ext == ".txt":
        try:
            text_content = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Failed to decode text file: {str(e)}")

        if not text_content.strip():
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Text file contains no readable text.")

        documents.append(
            Document(
                page_content=text_content,
                metadata={"document_id": doc_id, "filename": filename, "page": 1}
            )
        )

    elif ext == ".pdf":
        try:
            reader = pypdf.PdfReader(file_path)
            if len(reader.pages) == 0:
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="PDF file has no pages.")

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={"document_id": doc_id, "filename": filename, "page": i + 1}
                        )
                    )
        except Exception as e:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Unreadable or corrupted PDF file: {str(e)}")

        if not documents:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from PDF (it may be scanned/image-only or empty)."
            )

    # Chunking using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    split_chunks: List[Document] = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        split_chunks.extend(chunks)

    if not split_chunks:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed to create text chunks from document.")

    # Add chunks to FAISS vector store
    try:
        vector_store_manager.add_documents(split_chunks)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to vectorize and index document: {str(e)}")

    # Store metadata record
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    add_document_metadata(
        doc_id=doc_id,
        filename=filename,
        timestamp=timestamp,
        chunk_count=len(split_chunks),
        file_path=str(file_path)
    )

    return {
        "document_id": doc_id,
        "filename": filename,
        "chunk_count": len(split_chunks),
        "message": f"Successfully ingested '{filename}' with {len(split_chunks)} chunks."
    }
