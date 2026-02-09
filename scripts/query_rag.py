import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.huggingface import get_embedding_model
from src.vector_store.store import load_vector_db
from src.llm.model import get_llm
from src.retrieval.bridge import get_rag_chain

def main():
    print("Initializing RAG system...")
    embeddings = get_embedding_model()
    try:
        vector_db = load_vector_db(embeddings)
    except FileNotFoundError as e:
        print(e)
        return

    llm = get_llm()
    qa_chain = get_rag_chain(llm, vector_db)

    print("\n--- RAG System Ready ---")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
            
        print("Thinking...")
        try:
            response = qa_chain.invoke({"query": query})
            print(f"\nAnswer: {response['result']}")
            # print(f"\nSources: {[doc.metadata['source'] for doc in response['source_documents']]}")
        except Exception as e:
            print(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
