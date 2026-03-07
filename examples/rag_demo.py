"""
RAG Pipeline Demo - End-to-End Question Answering

This script demonstrates the complete RAG pipeline:
1. Load existing vector store from Phase 3
2. Initialize retriever and LLM
3. Create RAG chain
4. Run test queries with citation validation
5. Display results and metrics

Usage:
    python rag_demo.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store import FAISSVectorStore
from src.retrieval import SemanticRetriever
from src.llm.model import get_llm
from src.rag import RAGChain, RAGResponse
from src.utils.persistence import load_vector_store
from config.settings import (
    DB_DIR,
    MIN_SIMILARITY_THRESHOLD,
    MAX_CONTEXT_DOCS,
    RETRIEVAL_K
)


def print_header(text: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_response(query: str, response: RAGResponse, query_num: int):
    """Print formatted RAG response."""
    print(f"\n[Query {query_num}] {query}")
    print("-" * 80)
    
    # Answer
    print(f"\nAnswer:")
    print(f"  {response.answer}")
    
    # Confidence and status
    print(f"\nConfidence: {response.confidence:.4f}")
    print(f"Has Answer: {'Yes' if response.has_answer else 'No (Fallback)'}")
    
    # Citations
    if response.citations:
        print(f"\nCitations: {', '.join([f'[{c}]' for c in response.citations])}")
        
        # Citation validation
        validation = response.metadata.get('citation_validation', {})
        if validation:
            print(f"  - Valid citations: {len(validation.get('valid', []))}")
            print(f"  - Invalid citations: {len(validation.get('invalid', []))}")
            print(f"  - Citation coverage: {validation.get('coverage', 0):.1%}")
    else:
        print("\nCitations: None")
    
    # Sources
    if response.sources:
        print(f"\nSources ({len(response.sources)}):")
        for idx, source in enumerate(response.sources):
            text = source.get('text', source.get('page_content', ''))
            text_preview = text[:80].replace('\n', ' ')
            score = source.get('score', 0.0)
            print(f"  [{idx+1}] {text_preview}... (similarity: {score:.4f})")
    
    # Metadata
    query_time = response.metadata.get('query_time', 0)
    print(f"\nQuery Time: {query_time:.2f}s")


def main():
    """Run RAG pipeline demonstration."""
    
    print_header("RAG Pipeline Demo - Phase 4")
    print("\nThis demo shows end-to-end question answering with:")
    print("  - Semantic document retrieval")
    print("  - Strict context-only prompting")
    print("  - Citation extraction and validation")
    print("  - Confidence-based fallback handling")
    
    # Step 1: Load vector store
    print_header("Step 1: Loading Vector Store")
    try:
        vector_store = load_vector_store(DB_DIR, "rag_index")
        stats = vector_store.get_stats()
        print(f"\nLoaded vector store:")
        print(f"  - Total vectors: {stats['total_vectors']}")
        print(f"  - Dimension: {stats['dimension']}")
        print(f"  - Index type: {stats['index_type']}")
    except Exception as e:
        print(f"\nERROR: Failed to load vector store: {e}")
        print("\nPlease run pipeline_demo.py first to create the vector store.")
        return
    
    # Step 2: Initialize retriever
    print_header("Step 2: Initializing Semantic Retriever")
    retriever = SemanticRetriever(
        vector_store=vector_store,
        default_k=RETRIEVAL_K
    )
    print(f"\nRetriever configured:")
    print(f"  - Default k: {RETRIEVAL_K}")
    print(f"  - Vector store: {stats['total_vectors']} documents")
    
    # Step 3: Initialize LLM
    print_header("Step 3: Initializing Ollama LLM")
    try:
        llm = get_llm()
        print("\nOllama LLM initialized successfully")
        print("  - Model: tinyllama")
        print("  - Temperature: 0 (deterministic)")
    except Exception as e:
        print(f"\nERROR: Failed to initialize Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull model: ollama pull tinyllama")
        return
    
    # Step 4: Create RAG chain
    print_header("Step 4: Creating RAG Chain")
    rag_chain = RAGChain(
        retriever=retriever,
        llm=llm,
        min_similarity=MIN_SIMILARITY_THRESHOLD,
        max_docs=MAX_CONTEXT_DOCS
    )
    print(f"\nRAG Chain configured:")
    print(f"  - Min similarity threshold: {MIN_SIMILARITY_THRESHOLD}")
    print(f"  - Max context docs: {MAX_CONTEXT_DOCS}")
    print(f"  - Citation required: True")
    
    # Step 5: Run test queries
    print_header("Step 5: Running Test Queries")
    
    test_queries = [
        # Should find answers (from Phase 3 data)
        "What is machine learning?",
        "How to use Python for data analysis?",
        "What is HTML used for?",
        
        # Should return "no answer" (not in dataset)
        "What is quantum computing?",
        "Explain blockchain technology",
    ]
    
    print(f"\nExecuting {len(test_queries)} test queries...")
    
    results = []
    for idx, query in enumerate(test_queries, 1):
        response = rag_chain.query(query)
        results.append((query, response))
        print_response(query, response, idx)
    
    # Step 6: Summary statistics
    print_header("Step 6: Summary Statistics")
    
    total_queries = len(results)
    answered_queries = sum(1 for _, r in results if r.has_answer)
    fallback_queries = total_queries - answered_queries
    
    total_citations = sum(len(r.citations) for _, r in results)
    avg_confidence = sum(r.confidence for _, r in results) / total_queries
    avg_query_time = sum(r.metadata.get('query_time', 0) for _, r in results) / total_queries
    
    print(f"\nQuery Results:")
    print(f"  - Total queries: {total_queries}")
    print(f"  - Answered: {answered_queries} ({answered_queries/total_queries:.1%})")
    print(f"  - Fallback: {fallback_queries} ({fallback_queries/total_queries:.1%})")
    
    print(f"\nCitation Statistics:")
    print(f"  - Total citations: {total_citations}")
    print(f"  - Avg citations per query: {total_citations/total_queries:.1f}")
    
    print(f"\nPerformance:")
    print(f"  - Avg confidence: {avg_confidence:.4f}")
    print(f"  - Avg query time: {avg_query_time:.2f}s")
    
    # Step 7: Citation validation
    print_header("Step 7: Citation Validation")
    
    for idx, (query, response) in enumerate(results, 1):
        if response.citations:
            validation = response.metadata.get('citation_validation', {})
            valid = validation.get('valid', [])
            invalid = validation.get('invalid', [])
            coverage = validation.get('coverage', 0)
            
            status = "PASS" if validation.get('all_valid', False) else "FAIL"
            print(f"\n[Query {idx}] {status}")
            print(f"  Citations: {response.citations}")
            print(f"  Valid: {valid}")
            if invalid:
                print(f"  Invalid: {invalid}")
            print(f"  Coverage: {coverage:.1%}")
    
    # Final summary
    print_header("Demo Complete!")
    print("\nPhase 4 RAG pipeline successfully demonstrated:")
    print("  - Semantic retrieval working")
    print("  - LLM integration functional")
    print("  - Citations extracted and validated")
    print("  - Fallback handling operational")
    print("\nNext steps:")
    print("  - Run comprehensive tests: pytest test_rag.py")
    print("  - Integrate with main.py for interactive use")
    print("  - Add more documents to improve coverage")


if __name__ == "__main__":
    main()
