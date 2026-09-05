import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "app" / "storage"))
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "faiss_index"))
METADATA_FILE = STORAGE_DIR / "documents.json"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 4))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
