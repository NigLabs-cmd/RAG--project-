import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.huggingface import get_embedding_model
from src.vector_store.store import load_vector_db
from src.llm.model import get_llm
from src.retrieval.bridge import get_rag_chain

def main():
    print("Initializing RAG components...")
    try:
        embeddings = get_embedding_model()
        vector_db = load_vector_db(embeddings)
        llm = get_llm()
        qa_chain = get_rag_chain(llm, vector_db)
        
        query = "What is this document about?"
        print(f"Querying: {query}")
        
        response = qa_chain.invoke({"query": query})
        print(f"Result: {response['result']}")
        print("Test PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
