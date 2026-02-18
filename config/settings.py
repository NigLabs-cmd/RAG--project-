import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "vector_db")

# Models
## LLM
LLM_MODEL = "tinyllama"  # Smaller model for low-RAM systems (1.1B params)

## Embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 3  # Number of documents to retrieve

# RAG Chain Parameters
MIN_SIMILARITY_THRESHOLD = 0.2  # Lowered from 0.5 — FAISS inner-product scores for real PDFs are ~0.2-0.4
MAX_CONTEXT_DOCS = 3            # Limit context size for LLM
CITATION_REQUIRED = True        # Enforce citation extraction
LLM_TEMPERATURE = 0             # Deterministic for factual Q&A
