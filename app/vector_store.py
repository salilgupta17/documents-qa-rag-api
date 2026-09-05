import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import GOOGLE_API_KEY, FAISS_INDEX_DIR, METADATA_FILE, STORAGE_DIR

# Custom deterministic hash embedding fallback for offline/test mode if sentence-transformers download is unavailable
class OfflineHashEmbeddings(Embeddings):
    """Fallback embedding generator for offline testing when no external weights are available."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for i, char in enumerate(text):
            idx = (ord(char) + i * 31) % self.dimension
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_text(text)


def get_embeddings() -> Tuple[Embeddings, str]:
    """
    Returns appropriate embedding provider based on availability of GOOGLE_API_KEY.
    """
    if GOOGLE_API_KEY and GOOGLE_API_KEY.strip():
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            return embeddings, "gemini"
        except Exception as e:
            print(f"Warning: Failed to initialize GoogleGenerativeAIEmbeddings ({e}). Falling back to local offline embeddings.")
    
    return OfflineHashEmbeddings(), "offline"


class VectorStoreManager:
    def __init__(self):
        self.embeddings, self.mode = get_embeddings()
        self.vector_store: Optional[FAISS] = None
        self._load_or_init_index()

    def _load_or_init_index(self):
        index_file = FAISS_INDEX_DIR / "index.faiss"
        pkl_file = FAISS_INDEX_DIR / "index.pkl"
        if index_file.exists() and pkl_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(FAISS_INDEX_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Successfully loaded FAISS index from disk.")
            except Exception as e:
                print(f"Failed to load FAISS index: {e}. Starting with empty vector store.")
                self.vector_store = None
        else:
            self.vector_store = None

    def save_index(self):
        if self.vector_store is not None:
            FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(FAISS_INDEX_DIR))

    def add_documents(self, docs: List[Document]):
        if not docs:
            return
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)
        self.save_index()

    def delete_document_vectors(self, document_id: str) -> int:
        """Deletes all vectors associated with document_id from the FAISS index."""
        if self.vector_store is None:
            return 0

        # Find docstore_ids for chunks belonging to document_id
        docstore = self.vector_store.docstore
        ids_to_delete = []
        
        # docstore is an InMemoryDocstore
        for doc_id, doc in docstore._dict.items():
            if isinstance(doc, Document) and doc.metadata.get("document_id") == document_id:
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            try:
                self.vector_store.delete(ids_to_delete)
                # If index is now empty, reset vector store
                if len(self.vector_store.docstore._dict) == 0:
                    self.vector_store = None
                    # Clean up index files on disk
                    index_file = FAISS_INDEX_DIR / "index.faiss"
                    pkl_file = FAISS_INDEX_DIR / "index.pkl"
                    if index_file.exists():
                        index_file.unlink()
                    if pkl_file.exists():
                        pkl_file.unlink()
                else:
                    self.save_index()
            except Exception as e:
                print(f"Error deleting vectors from FAISS: {e}")
                # Fallback: rebuild index without the deleted document
                self._rebuild_index_without(document_id)
        
        return len(ids_to_delete)

    def _rebuild_index_without(self, document_id: str):
        """Fallback method to rebuild FAISS index excluding a document_id."""
        if self.vector_store is None:
            return
        remaining_docs = [
            doc for doc in self.vector_store.docstore._dict.values()
            if isinstance(doc, Document) and doc.metadata.get("document_id") != document_id
        ]
        if remaining_docs:
            self.vector_store = FAISS.from_documents(remaining_docs, self.embeddings)
            self.save_index()
        else:
            self.vector_store = None
            index_file = FAISS_INDEX_DIR / "index.faiss"
            pkl_file = FAISS_INDEX_DIR / "index.pkl"
            if index_file.exists():
                index_file.unlink()
            if pkl_file.exists():
                pkl_file.unlink()

    def similarity_search(self, query: str, k: int = 4, document_id: Optional[str] = None) -> List[Document]:
        if self.vector_store is None:
            return []
        
        filter_dict = {"document_id": document_id} if document_id else None
        
        try:
            return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
        except Exception as e:
            # Fallback if filtering isn't directly supported by specific FAISS backend setup
            docs = self.vector_store.similarity_search(query, k=k * 3)
            if document_id:
                docs = [d for d in docs if d.metadata.get("document_id") == document_id]
            return docs[:k]

    def get_total_chunks(self) -> int:
        if self.vector_store is None:
            return 0
        return len(self.vector_store.docstore._dict)


# Metadata helpers
def load_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_metadata(data: Dict[str, Dict[str, Any]]):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def add_document_metadata(doc_id: str, filename: str, timestamp: str, chunk_count: int, file_path: str):
    data = load_metadata()
    data[doc_id] = {
        "document_id": doc_id,
        "filename": filename,
        "upload_timestamp": timestamp,
        "chunk_count": chunk_count,
        "file_path": file_path
    }
    save_metadata(data)

def remove_document_metadata(doc_id: str) -> Optional[Dict[str, Any]]:
    data = load_metadata()
    doc_info = data.pop(doc_id, None)
    if doc_info:
        save_metadata(data)
    return doc_info

def get_all_documents() -> List[Dict[str, Any]]:
    data = load_metadata()
    return list(data.values())

# Global singleton manager instance
vector_store_manager = VectorStoreManager()
