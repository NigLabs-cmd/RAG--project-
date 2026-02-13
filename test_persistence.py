"""
Test suite for persistence layer utilities.

Tests FAISS vector store persistence functions including:
- Save/load vector store
- Statistics saving
- Validation
- Info retrieval
- Backup creation
"""

import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store.faiss_store import FAISSVectorStore
from src.utils.persistence import (
    save_vector_store,
    load_vector_store,
    save_vector_store_stats,
    validate_vector_store,
    get_vector_store_info,
    create_vector_store_backup
)
from src.embeddings.huggingface import get_embedding_model, EXPECTED_EMBEDDING_DIM
import numpy as np


def create_test_vector_store():
    """Create a test vector store with sample data."""
    # Create sample embeddings
    n_samples = 20
    embeddings = np.random.randn(n_samples, EXPECTED_EMBEDDING_DIM).tolist()
    
    metadatas = [
        {"doc_id": i, "category": f"cat_{i % 3}", "source": f"doc_{i}.txt"}
        for i in range(n_samples)
    ]
    
    texts = [f"Sample document {i} with some content" for i in range(n_samples)]
    
    # Create vector store
    store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    store.add_embeddings(embeddings, metadatas, texts)
    
    return store


def test_save_vector_store():
    """Test saving vector store with utilities."""
    print("\n" + "="*60)
    print("TEST 1: Save Vector Store")
    print("="*60)
    
    test_dir = "test_persistence_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create and save vector store
        store = create_test_vector_store()
        
        saved_files = save_vector_store(
            store,
            test_dir,
            index_name="test_index",
            include_stats=True
        )
        
        # Verify files were created
        assert "index" in saved_files
        assert "metadata" in saved_files
        assert "stats" in saved_files
        
        assert os.path.exists(saved_files["index"])
        assert os.path.exists(saved_files["metadata"])
        assert os.path.exists(saved_files["stats"])
        
        print(f"OK: Saved {len(saved_files)} files")
        for key, path in saved_files.items():
            print(f"  {key}: {path}")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_load_vector_store():
    """Test loading vector store with utilities."""
    print("\n" + "="*60)
    print("TEST 2: Load Vector Store")
    print("="*60)
    
    test_dir = "test_persistence_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create and save
        store1 = create_test_vector_store()
        save_vector_store(store1, test_dir, "test_index")
        
        # Load
        store2 = load_vector_store(
            test_dir,
            "test_index",
            dimension=EXPECTED_EMBEDDING_DIM,
            validate=True
        )
        
        # Verify loaded correctly
        assert store2.index.ntotal == store1.index.ntotal
        assert len(store2.metadata_map) == len(store1.metadata_map)
        assert store2.dimension == store1.dimension
        
        print(f"OK: Loaded vector store with {store2.index.ntotal} vectors")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_validate_vector_store():
    """Test vector store validation."""
    print("\n" + "="*60)
    print("TEST 3: Validate Vector Store")
    print("="*60)
    
    # Test valid store
    store = create_test_vector_store()
    is_valid, issues = validate_vector_store(store)
    
    assert is_valid
    assert len(issues) == 0
    print("OK: Valid vector store passed validation")
    
    # Test empty store
    empty_store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    is_valid, issues = validate_vector_store(empty_store)
    
    assert not is_valid  # Empty store should have issues
    print(f"OK: Empty store detected ({len(issues)} issues)")


def test_get_vector_store_info():
    """Test getting vector store info without loading."""
    print("\n" + "="*60)
    print("TEST 4: Get Vector Store Info")
    print("="*60)
    
    test_dir = "test_persistence_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create and save
        store = create_test_vector_store()
        save_vector_store(store, test_dir, "test_index", include_stats=True)
        
        # Get info without loading
        info = get_vector_store_info(test_dir, "test_index")
        
        assert info["files_exist"]["index"]
        assert info["files_exist"]["metadata"]
        assert info["files_exist"]["stats"]
        assert info["metadata"] is not None
        assert info["stats"] is not None
        
        print("OK: Retrieved vector store info")
        print(f"  Dimension: {info['metadata']['dimension']}")
        print(f"  Total vectors: {info['metadata']['total_vectors']}")
        print(f"  Created: {info['metadata']['created_at']}")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_create_backup():
    """Test creating vector store backup."""
    print("\n" + "="*60)
    print("TEST 5: Create Backup")
    print("="*60)
    
    test_dir = "test_persistence_temp"
    backup_dir = "test_backup_temp"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    
    try:
        # Create and save
        store = create_test_vector_store()
        save_vector_store(store, test_dir, "test_index", include_stats=True)
        
        # Create backup
        backup_path = create_vector_store_backup(
            test_dir,
            backup_dir,
            "test_index"
        )
        
        assert os.path.exists(backup_path)
        
        # Verify backup files
        backup_files = os.listdir(backup_path)
        assert len(backup_files) >= 2  # At least index and metadata
        
        print(f"OK: Created backup at: {backup_path}")
        print(f"  Files backed up: {len(backup_files)}")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)


def test_save_load_cycle():
    """Test complete save/load cycle preserves data."""
    print("\n" + "="*60)
    print("TEST 6: Save/Load Cycle")
    print("="*60)
    
    test_dir = "test_persistence_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create original store
        store1 = create_test_vector_store()
        original_stats = store1.get_statistics()
        
        # Save
        save_vector_store(store1, test_dir, "test_index")
        
        # Load
        store2 = load_vector_store(test_dir, "test_index")
        loaded_stats = store2.get_statistics()
        
        # Verify statistics match
        assert loaded_stats["total_vectors"] == original_stats["total_vectors"]
        assert loaded_stats["dimension"] == original_stats["dimension"]
        assert loaded_stats["total_metadata"] == original_stats["total_metadata"]
        
        # Test search works on loaded store
        query_emb = np.random.randn(EXPECTED_EMBEDDING_DIM).tolist()
        results = store2.search(query_emb, k=5)
        
        assert len(results) == 5
        
        print("OK: Save/load cycle preserves data")
        print(f"  Vectors: {loaded_stats['total_vectors']}")
        print(f"  Search works: {len(results)} results")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PERSISTENCE LAYER TEST SUITE")
    print("="*60)
    
    try:
        test_save_vector_store()
        test_load_vector_store()
        test_validate_vector_store()
        test_get_vector_store_info()
        test_create_backup()
        test_save_load_cycle()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nPersistence Features Verified:")
        print("  OK: Save vector store with stats")
        print("  OK: Load vector store with validation")
        print("  OK: Validate vector store integrity")
        print("  OK: Get info without loading")
        print("  OK: Create backups")
        print("  OK: Complete save/load cycle")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
