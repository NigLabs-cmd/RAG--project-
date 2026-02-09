import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATA_DIR
from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents
from src.embeddings.huggingface import get_embedding_model
from src.vector_store.store import create_vector_db

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}. Please add PDF or TXT files there.")
        return

    documents = []
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        try:
            doc = load_document(file_path)
            documents.extend(doc)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not documents:
        print("No documents found or loaded.")
        return

    chunks = split_documents(documents)
    embeddings = get_embedding_model()
    create_vector_db(chunks, embeddings)

if __name__ == "__main__":
    main()
