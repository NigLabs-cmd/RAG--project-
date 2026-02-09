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
