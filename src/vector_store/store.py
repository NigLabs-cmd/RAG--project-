import os
from langchain_community.vectorstores import FAISS
from config.settings import DB_DIR

def create_vector_db(chunks, embedding_model):
    """
    Creates a FAISS vector database from document chunks and saves it locally.
    """
    if not chunks:
        print("No chunks to process.")
        return None

    print("Creating FAISS index...")
    vector_db = FAISS.from_documents(chunks, embedding_model)
    
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        
    vector_db.save_local(DB_DIR)
    print(f"Vector DB saved to {DB_DIR}")
    return vector_db

def load_vector_db(embedding_model):
    """
    Loads the FAISS vector database from local storage.
    """
    if not os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        raise FileNotFoundError(f"FAISS index not found in {DB_DIR}. Run ingestion first.")
        
    print("Loading FAISS index...")
    vector_db = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True) 
    # allow_dangerous_deserialization is needed for local files we trust
    return vector_db
