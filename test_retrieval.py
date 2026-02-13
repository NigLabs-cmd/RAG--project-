"""
Comprehensive test suite for Semantic Retrieval System.

Tests:
- Query embedding generation
- Top-k retrieval
- Similarity score tracking
- Batch retrieval
- Result formatting (dict and Document)
- Known query-answer pairs
- Performance benchmarking
- Edge cases
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import SemanticRetriever, create_retriever, retrieve_context
from src.embeddings.huggingface import get_embedding_model, EXPECTED_EMBEDDING_DIM


def create_sample_vector_store():
    """Create a sample vector store with test data."""
    # Sample documents about different topics
    documents = [
        # Machine Learning (0-2)
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Supervised learning uses labeled data to train models for prediction tasks.",
        "Neural networks are computational models inspired by the human brain structure.",
        
        # Python Programming (3-5)
        "Python is a high-level programming language known for its simplicity and readability.",
        "List comprehensions in Python provide a concise way to create lists.",
        "The pandas library is essential for data manipulation and analysis in Python.",
        
        # Data Science (6-8)
        "Data science combines statistics, programming, and domain expertise to extract insights.",
        "Feature engineering is the process of creating new variables from existing data.",
        "Cross-validation helps assess model performance and prevent overfitting.",
        
        # Web Development (9-11)
        "HTML is the standard markup language for creating web pages.",
        "CSS is used to style and layout web pages with colors, fonts, and spacing.",
        "JavaScript enables interactive and dynamic content on websites.",
        
        # Databases (12-14)
        "SQL is a language for managing and querying relational databases.",
        "NoSQL databases like MongoDB are designed for unstructured data storage.",
        "Database indexing improves query performance by creating efficient lookup structures."
    ]
    
    # Create metadata
    categories = ["ML", "ML", "ML", "Python", "Python", "Python", 
                  "DataScience", "DataScience", "DataScience",
                  "Web", "Web", "Web", "Database", "Database", "Database"]
    
    metadatas = [
        {"category": cat, "doc_id": i, "source": f"doc_{i}.txt"}
        for i, cat in enumerate(categories)
    ]
    
    # Generate embeddings
    print("Generating embeddings for sample documents...")
    model = get_embedding_model()
    embeddings = model.embed_documents(documents)
    
    # Create vector store
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    store.add_embeddings(embeddings, metadatas, documents)
    
    print(f"Created vector store with {len(documents)} documents")
    
    return store, documents


def test_retriever_initialization():
    """Test retriever initialization."""
    print("\n" + "="*60)
    print("TEST 1: Retriever Initialization")
    print("="*60)
    
    store, _ = create_sample_vector_store()
    
    # Test with default parameters
    retriever = SemanticRetriever(store)
    assert retriever.vector_store == store
    assert retriever.embedding_model is not None
    assert retriever.default_k == 5
    
    print("OK: Default retriever initialized")
    
    # Test with custom parameters
    retriever2 = SemanticRetriever(store, default_k=10, default_min_score=0.5)
    assert retriever2.default_k == 10
    assert retriever2.default_min_score == 0.5
    
    print("OK: Custom retriever initialized")
    
    # Test convenience function
    retriever3 = create_retriever(store, k=3)
    assert retriever3.default_k == 3
    
    print("OK: Convenience function works")
    
    return store


def test_single_query_retrieval(store):
    """Test single query retrieval."""
    print("\n" + "="*60)
    print("TEST 2: Single Query Retrieval")
    print("="*60)
    
    retriever = SemanticRetriever(store, default_k=3)
    
    # Test query about machine learning
    query = "What is machine learning?"
    results = retriever.retrieve(query)
    
    assert len(results) <= 3
    assert all("score" in r for r in results)
    assert all("text" in r for r in results)
    assert all("metadata" in r for r in results)
    
    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. [Score: {result['score']:.4f}] {result['text'][:80]}...")
        print(f"     Category: {result['metadata'].get('category', 'N/A')}")
    
    # Verify top result is about ML
    assert "machine learning" in results[0]["text"].lower()
    print("\nOK: Top result is relevant to query")


def test_top_k_parameter(store):
    """Test different k values."""
    print("\n" + "="*60)
    print("TEST 3: Top-K Parameter")
    print("="*60)
    
    retriever = SemanticRetriever(store)
    query = "Tell me about Python programming"
    
    # Test different k values
    for k in [1, 3, 5, 10]:
        results = retriever.retrieve(query, k=k)
        print(f"k={k}: Retrieved {len(results)} results")
        assert len(results) <= k
    
    print("OK: Top-k parameter works correctly")


def test_min_score_filtering(store):
    """Test minimum score filtering."""
    print("\n" + "="*60)
    print("TEST 4: Minimum Score Filtering")
    print("="*60)
    
    retriever = SemanticRetriever(store)
    query = "What is data science?"
    
    # Without filtering
    results_all = retriever.retrieve(query, k=10)
    print(f"Without filtering: {len(results_all)} results")
    
    # With high threshold
    results_filtered = retriever.retrieve(query, k=10, min_score=0.6)
    print(f"With min_score=0.6: {len(results_filtered)} results")
    
    # Verify all scores are above threshold
    for result in results_filtered:
        assert result["score"] >= 0.6
    
    assert len(results_filtered) <= len(results_all)
    print("OK: Score filtering works correctly")


def test_batch_retrieval(store):
    """Test batch query retrieval."""
    print("\n" + "="*60)
    print("TEST 5: Batch Retrieval")
    print("="*60)
    
    retriever = SemanticRetriever(store, default_k=3)
    
    queries = [
        "What is machine learning?",
        "How to use Python?",
        "What is a database?"
    ]
    
    batch_results = retriever.batch_retrieve(queries)
    
    assert len(batch_results) == len(queries)
    assert all(len(results) <= 3 for results in batch_results)
    
    print(f"Processed {len(queries)} queries in batch")
    for i, (query, results) in enumerate(zip(queries, batch_results)):
        print(f"\nQuery {i+1}: '{query}'")
        print(f"  Top result: {results[0]['text'][:60]}...")
        print(f"  Score: {results[0]['score']:.4f}")
    
    print("\nOK: Batch retrieval works correctly")


def test_document_format(store):
    """Test returning results as LangChain Documents."""
    print("\n" + "="*60)
    print("TEST 6: Document Format")
    print("="*60)
    
    retriever = SemanticRetriever(store, default_k=3)
    query = "What is neural networks?"
    
    # Get results as Documents
    documents = retriever.retrieve(query, return_documents=True)
    
    assert len(documents) <= 3
    
    # Verify Document structure
    for doc in documents:
        assert hasattr(doc, 'page_content')
        assert hasattr(doc, 'metadata')
        assert 'retrieval_score' in doc.metadata
        assert 'retrieval_rank' in doc.metadata
    
    print(f"Retrieved {len(documents)} LangChain Documents")
    print(f"First document content: {documents[0].page_content[:80]}...")
    print(f"Metadata keys: {list(documents[0].metadata.keys())}")
    
    print("OK: Document format works correctly")


def test_known_queries(store):
    """Test retrieval accuracy with known query-answer pairs."""
    print("\n" + "="*60)
    print("TEST 7: Known Query-Answer Pairs")
    print("="*60)
    
    retriever = SemanticRetriever(store, default_k=3)
    
    # Define known query-answer pairs
    test_cases = [
        {
            "query": "What is machine learning?",
            "expected_category": "ML",
            "expected_keywords": ["machine learning", "artificial intelligence"]
        },
        {
            "query": "How to use Python for data analysis?",
            "expected_category": "Python",
            "expected_keywords": ["python", "pandas", "data"]
        },
        {
            "query": "What is HTML used for?",
            "expected_category": "Web",
            "expected_keywords": ["html", "web"]
        }
    ]
    
    passed = 0
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        results = retriever.retrieve(query, k=3)
        
        # Check if top result matches expected category
        top_category = results[0]["metadata"].get("category", "")
        top_text = results[0]["text"].lower()
        
        # Check if any expected keyword is in top result
        has_keyword = any(kw.lower() in top_text for kw in test_case["expected_keywords"])
        
        if has_keyword:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"\nTest {i+1}: {status}")
        print(f"  Query: {query}")
        print(f"  Expected category: {test_case['expected_category']}")
        print(f"  Got category: {top_category}")
        print(f"  Top result: {results[0]['text'][:60]}...")
        print(f"  Score: {results[0]['score']:.4f}")
    
    accuracy = (passed / len(test_cases)) * 100
    print(f"\nAccuracy: {passed}/{len(test_cases)} ({accuracy:.1f}%)")
    
    assert passed >= len(test_cases) * 0.6  # At least 60% accuracy
    print("OK: Retrieval accuracy is acceptable")


def test_performance(store):
    """Test retrieval performance."""
    print("\n" + "="*60)
    print("TEST 8: Performance Benchmarking")
    print("="*60)
    
    retriever = SemanticRetriever(store, default_k=5)
    
    queries = [
        "What is machine learning?",
        "How to program in Python?",
        "What is a database?",
        "Tell me about web development",
        "What is data science?"
    ]
    
    # Measure retrieval time
    times = []
    for query in queries:
        start = time.time()
        results = retriever.retrieve(query)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Queries processed: {len(queries)}")
    print(f"Average time: {avg_time:.2f}ms")
    print(f"Min time: {min_time:.2f}ms")
    print(f"Max time: {max_time:.2f}ms")
    
    # Performance should be reasonable (< 500ms per query)
    assert avg_time < 500
    print("OK: Performance is acceptable")


def test_retrieval_statistics(store):
    """Test retrieval statistics."""
    print("\n" + "="*60)
    print("TEST 9: Retrieval Statistics")
    print("="*60)
    
    retriever = SemanticRetriever(store, default_k=5, default_min_score=0.3)
    
    stats = retriever.get_retrieval_statistics()
    
    assert "vector_store" in stats
    assert "default_k" in stats
    assert "default_min_score" in stats
    assert stats["default_k"] == 5
    assert stats["default_min_score"] == 0.3
    
    print("Retrieval Statistics:")
    for key, value in stats.items():
        if key != "vector_store":
            print(f"  {key}: {value}")
    
    print("OK: Statistics retrieved successfully")


def test_edge_cases(store):
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 10: Edge Cases")
    print("="*60)
    
    retriever = SemanticRetriever(store)
    
    # Test empty query
    results = retriever.retrieve("")
    assert len(results) == 0
    print("OK: Empty query handled")
    
    # Test whitespace-only query
    results = retriever.retrieve("   ")
    assert len(results) == 0
    print("OK: Whitespace query handled")
    
    # Test very long query
    long_query = "machine learning " * 100
    results = retriever.retrieve(long_query, k=3)
    assert len(results) <= 3
    print("OK: Long query handled")
    
    # Test k=0 (may return 0 or some results depending on FAISS behavior)
    results = retriever.retrieve("test query", k=0)
    # FAISS may still return results even with k=0, so just check it doesn't crash
    print(f"OK: k=0 handled (returned {len(results)} results)")


def test_retrieve_context_function():
    """Test the convenience retrieve_context function."""
    print("\n" + "="*60)
    print("TEST 11: Retrieve Context Function")
    print("="*60)
    
    store, _ = create_sample_vector_store()
    
    # Test as documents
    docs = retrieve_context("What is Python?", store, k=3)
    assert len(docs) <= 3
    assert all(hasattr(doc, 'page_content') for doc in docs)
    print(f"OK: Retrieved {len(docs)} documents")
    
    # Test as string
    context_str = retrieve_context("What is Python?", store, k=3, as_string=True)
    assert isinstance(context_str, str)
    assert len(context_str) > 0
    print(f"OK: Retrieved context string ({len(context_str)} chars)")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SEMANTIC RETRIEVAL TEST SUITE")
    print("="*60)
    
    try:
        # Create sample data once
        store = test_retriever_initialization()
        
        # Run all tests
        test_single_query_retrieval(store)
        test_top_k_parameter(store)
        test_min_score_filtering(store)
        test_batch_retrieval(store)
        test_document_format(store)
        test_known_queries(store)
        test_performance(store)
        test_retrieval_statistics(store)
        test_edge_cases(store)
        test_retrieve_context_function()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nRetrieval System Features Verified:")
        print("  OK: Retriever initialization")
        print("  OK: Single query retrieval")
        print("  OK: Top-k parameter configuration")
        print("  OK: Minimum score filtering")
        print("  OK: Batch retrieval")
        print("  OK: Document format conversion")
        print("  OK: Known query accuracy")
        print("  OK: Performance benchmarking")
        print("  OK: Statistics tracking")
        print("  OK: Edge case handling")
        print("  OK: Convenience functions")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
