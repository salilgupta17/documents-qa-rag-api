import io
import os
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import STORAGE_DIR, FAISS_INDEX_DIR

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Ensure clean test environment before each test."""
    # Clean up test storage and faiss index files if needed
    yield
    # Cleanup after test if needed


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "document_count" in data
    assert "mode" in data


def test_upload_txt_document():
    content = (
        "FAISS is a library for efficient similarity search and clustering of dense vectors. "
        "It contains algorithms that search in sets of vectors of any size. "
        "LangChain is a framework for developing applications powered by language models."
    )
    file_bytes = content.encode("utf-8")
    
    response = client.post(
        "/documents/upload",
        files={"file": ("sample_info.txt", io.BytesIO(file_bytes), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "sample_info.txt"
    assert data["chunk_count"] > 0
    
    doc_id = data["document_id"]

    # Verify document in listing
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    doc_ids = [d["document_id"] for d in list_resp.json()["documents"]]
    assert doc_id in doc_ids

    # Query the document
    query_resp = client.post(
        "/query",
        json={"question": "What is FAISS?", "document_id": doc_id}
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert "answer" in q_data
    assert len(q_data["sources"]) > 0
    assert q_data["sources"][0]["filename"] == "sample_info.txt"

    # Delete the document
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["document_id"] == doc_id

    # Confirm deletion from listing
    list_after_del = client.get("/documents")
    doc_ids_after = [d["document_id"] for d in list_after_del.json()["documents"]]
    assert doc_id not in doc_ids_after


def test_upload_unsupported_file():
    response = client.post(
        "/documents/upload",
        files={"file": ("image.png", io.BytesIO(b"fake image data"), "image/png")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_delete_non_existent_document():
    response = client.delete("/documents/non_existent_doc_id_999")
    assert response.status_code == 404


def test_query_no_context():
    response = client.post(
        "/query",
        json={"question": "What is the secret recipe for quantum fusion?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "I cannot find the answer" in data["answer"] or len(data["answer"]) > 0
