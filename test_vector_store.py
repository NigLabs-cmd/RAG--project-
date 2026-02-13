"""
Comprehensive test suite for FAISS Vector Store implementation.

Tests:
- Index initialization
- Adding embeddings
- Metadata mapping
- Search functionality
- Save/load persistence
- Statistics and validation
- Edge cases
"""

import sys
import os
import shutil
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store.faiss_store import FAISSVectorStore
from src.embeddings.huggingface import get_embedding_model, EXPECTED_EMBEDDING_DIM


def test_initialization():
    """Test vector store initialization."""
    print("\n" + "="*60)
    print("TEST 1: Vector Store Initialization")
    print("="*60)
    
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    
    assert store.dimension == EXPECTED_EMBEDDING_DIM
    assert store.index is None  # Index created lazily
    assert store.id_counter == 0
    assert len(store.metadata_map) == 0
    
    print(f"✓ Vector store initialized: {store}")
    print(f"✓ Dimension: {store.dimension}")
    

def test_add_embeddings():
    """Test adding embeddings to the index."""
    print("\n" + "="*60)
    print("TEST 2: Adding Embeddings")
    print("="*60)
    
    # Create sample embeddings
    n_samples = 10
    embeddings = np.random.randn(n_samples, EXPECTED_EMBEDDING_DIM).tolist()
    
    # Create metadata
    metadatas = [
        {"source": f"doc_{i}", "page": i % 3, "chunk_id": i}
        for i in range(n_samples)
    ]
    
    # Create texts
    texts = [f"This is sample text chunk number {i}" for i in range(n_samples)]
    
    # Add to store
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    ids = store.add_embeddings(embeddings, metadatas, texts)
    
    assert len(ids) == n_samples
    assert store.index is not None
    assert store.index.ntotal == n_samples
    assert len(store.metadata_map) == n_samples
    
    print(f"✓ Added {n_samples} embeddings")
    print(f"✓ Assigned IDs: {ids}")
    print(f"✓ Index size: {store.index.ntotal}")
    print(f"✓ Metadata entries: {len(store.metadata_map)}")
    
    # Verify metadata
    for i, idx in enumerate(ids):
        metadata = store.metadata_map[idx]
        assert metadata["id"] == idx
        assert metadata["source"] == f"doc_{i}"
        assert metadata["text"] == texts[i]
    
    print("✓ Metadata mapping verified")
    
    return store, embeddings, metadatas, texts


def test_search():
    """Test search functionality."""
    print("\n" + "="*60)
    print("TEST 3: Search Functionality")
    print("="*60)
    
    # Create store with embeddings
    n_samples = 20
    embeddings = np.random.randn(n_samples, EXPECTED_EMBEDDING_DIM).tolist()
    metadatas = [{"doc_id": i} for i in range(n_samples)]
    texts = [f"Document {i}" for i in range(n_samples)]
    
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    store.add_embeddings(embeddings, metadatas, texts)
    
    # Test search with first embedding as query
    query_embedding = embeddings[0]
    results = store.search(query_embedding, k=5)
    
    assert len(results) <= 5
    assert results[0]["id"] == 0  # Should find itself first
    assert results[0]["score"] > 0.99  # Should be very similar to itself
    
    print(f"✓ Search returned {len(results)} results")
    print(f"✓ Top result ID: {results[0]['id']}, Score: {results[0]['score']:.4f}")
    
    # Test with minimum score filter
    results_filtered = store.search(query_embedding, k=5, min_score=0.5)
    
    print(f"✓ Filtered search (min_score=0.5): {len(results_filtered)} results")
    
    # Verify all scores are above threshold
    for result in results_filtered:
        assert result["score"] >= 0.5
    
    print("✓ Score filtering works correctly")
    
    return store


def test_batch_search():
    """Test batch search functionality."""
    print("\n" + "="*60)
    print("TEST 4: Batch Search")
    print("="*60)
    
    # Create store
    n_samples = 15
    embeddings = np.random.randn(n_samples, EXPECTED_EMBEDDING_DIM).tolist()
    
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    store.add_embeddings(embeddings)
    
    # Batch search with multiple queries
    query_embeddings = [embeddings[0], embeddings[5], embeddings[10]]
    batch_results = store.batch_search(query_embeddings, k=3)
    
    assert len(batch_results) == 3
    assert all(len(results) <= 3 for results in batch_results)
    
    print(f"✓ Batch search for {len(query_embeddings)} queries")
    print(f"✓ Results per query: {[len(r) for r in batch_results]}")


def test_statistics():
    """Test statistics functionality."""
    print("\n" + "="*60)
    print("TEST 5: Statistics")
    print("="*60)
    
    # Create store with data
    n_samples = 25
    embeddings = np.random.randn(n_samples, EXPECTED_EMBEDDING_DIM).tolist()
    
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    store.add_embeddings(embeddings)
    
    stats = store.get_statistics()
    
    assert stats["dimension"] == EXPECTED_EMBEDDING_DIM
    assert stats["total_vectors"] == n_samples
    assert stats["total_metadata"] == n_samples
    assert stats["index_type"] == "IndexFlatIP (Cosine Similarity)"
    assert stats["is_trained"] == True
    
    print("✓ Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_save_load():
    """Test save and load functionality."""
    print("\n" + "="*60)
    print("TEST 6: Save and Load")
    print("="*60)
    
    # Create temporary directory for testing
    test_dir = "test_vector_store_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create and populate store
        n_samples = 30
        embeddings = np.random.randn(n_samples, EXPECTED_EMBEDDING_DIM).tolist()
        metadatas = [{"doc_id": i, "category": f"cat_{i % 3}"} for i in range(n_samples)]
        texts = [f"Sample text {i}" for i in range(n_samples)]
        
        store1 = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
        ids = store1.add_embeddings(embeddings, metadatas, texts)
        
        # Save
        store1.save(test_dir, "test_index")
        print(f"✓ Saved vector store to: {test_dir}")
        
        # Load into new store
        store2 = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
        store2.load(test_dir, "test_index")
        print(f"✓ Loaded vector store from: {test_dir}")
        
        # Verify loaded data
        assert store2.dimension == store1.dimension
        assert store2.index.ntotal == store1.index.ntotal
        assert len(store2.metadata_map) == len(store1.metadata_map)
        assert store2.id_counter == store1.id_counter
        
        print(f"✓ Loaded {store2.index.ntotal} vectors")
        print(f"✓ Loaded {len(store2.metadata_map)} metadata entries")
        
        # Verify search works on loaded store
        query_embedding = embeddings[0]
        results = store2.search(query_embedding, k=5)
        
        assert len(results) > 0
        assert results[0]["id"] == 0
        
        print("✓ Search works on loaded store")
        print(f"✓ Top result: ID={results[0]['id']}, Score={results[0]['score']:.4f}")
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"✓ Cleaned up test directory")


def test_real_embeddings():
    """Test with real embeddings from the model."""
    print("\n" + "="*60)
    print("TEST 7: Real Embeddings")
    print("="*60)
    
    # Load embedding model
    model = get_embedding_model()
    
    # Sample texts
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "The weather is sunny today.",
        "I like to eat pizza for dinner."
    ]
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.embed_documents(texts)
    
    # Create store
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    metadatas = [{"text_id": i, "category": "test"} for i in range(len(texts))]
    ids = store.add_embeddings(embeddings, metadatas, texts)
    
    print(f"✓ Added {len(ids)} real embeddings to store")
    
    # Test semantic search
    query = "What is machine learning?"
    print(f"\nQuery: '{query}'")
    
    query_embedding = model.embed_query(query)
    results = store.search(query_embedding, k=3)
    
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['text']}")
    
    # The first result should be about machine learning
    assert "machine learning" in results[0]["text"].lower()
    print("\n✓ Semantic search works correctly!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 8: Edge Cases")
    print("="*60)
    
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    
    # Test search on empty index
    query_embedding = np.random.randn(EXPECTED_EMBEDDING_DIM).tolist()
    results = store.search(query_embedding, k=5)
    assert len(results) == 0
    print("✓ Empty index search returns empty results")
    
    # Test adding empty embeddings
    ids = store.add_embeddings([])
    assert len(ids) == 0
    print("✓ Adding empty embeddings handled correctly")
    
    # Add some data
    embeddings = np.random.randn(5, EXPECTED_EMBEDDING_DIM).tolist()
    store.add_embeddings(embeddings)
    
    # Test k larger than index size
    results = store.search(query_embedding, k=100)
    assert len(results) <= 5
    print("✓ k larger than index size handled correctly")
    
    # Test dimension mismatch
    try:
        wrong_dim_embedding = np.random.randn(256).tolist()
        store.add_embeddings([wrong_dim_embedding])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Dimension mismatch caught: {e}")
    
    # Test clear
    store.clear()
    assert store.index is None
    assert len(store.metadata_map) == 0
    print("✓ Clear works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FAISS VECTOR STORE TEST SUITE")
    print("=" * 60)
    
    try:
        test_initialization()
        test_add_embeddings()
        test_search()
        test_batch_search()
        test_statistics()
        test_save_load()
        test_real_embeddings()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nVector Store Features Verified:")
        print("  ✓ Index initialization")
        print("  ✓ Adding embeddings with metadata")
        print("  ✓ Single and batch search")
        print("  ✓ Statistics tracking")
        print("  ✓ Save/load persistence")
        print("  ✓ Real embedding integration")
        print("  ✓ Edge case handling")
        print("  ✓ Semantic search accuracy")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
