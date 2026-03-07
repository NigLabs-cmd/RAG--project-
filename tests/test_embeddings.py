"""
Test script for embedding validation functionality.
Tests validate_embeddings(), log_embedding_statistics(), and embedding generation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.huggingface import (
    get_embedding_model, 
    validate_embeddings, 
    log_embedding_statistics,
    EXPECTED_EMBEDDING_DIM
)
import numpy as np


def test_embedding_model_loading():
    """Test that the embedding model loads correctly."""
    print("\n" + "="*60)
    print("TEST 1: Embedding Model Loading")
    print("="*60)
    
    model = get_embedding_model()
    assert model is not None, "Model should load successfully"
    print("✓ Embedding model loaded successfully")
    
    return model


def test_embedding_generation(model):
    """Test generating embeddings for sample texts."""
    print("\n" + "="*60)
    print("TEST 2: Embedding Generation")
    print("="*60)
    
    # Sample texts
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language."
    ]
    
    print(f"\nGenerating embeddings for {len(texts)} text samples...")
    
    # Generate embeddings
    embeddings = model.embed_documents(texts)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Basic checks
    assert len(embeddings) == len(texts), "Should have one embedding per text"
    assert all(len(emb) == EXPECTED_EMBEDDING_DIM for emb in embeddings), \
        f"All embeddings should have dimension {EXPECTED_EMBEDDING_DIM}"
    
    print(f"✓ All embeddings have correct dimension: {EXPECTED_EMBEDDING_DIM}")
    
    return embeddings, texts


def test_validate_embeddings_function(embeddings, texts):
    """Test the validate_embeddings function."""
    print("\n" + "="*60)
    print("TEST 3: Embedding Validation Function")
    print("="*60)
    
    # Test valid embeddings
    is_valid, reason, details = validate_embeddings(embeddings, texts)
    
    print(f"\nValidation result: {is_valid}")
    print(f"Reason: {reason}")
    print(f"\nValidation details:")
    for key, value in details.items():
        if key != "dimension_mismatch" or value:  # Only show if there are mismatches
            print(f"  {key}: {value}")
    
    assert is_valid, f"Valid embeddings should pass validation: {reason}"
    print("\n✓ Valid embeddings passed validation")
    
    # Test edge cases
    print("\n" + "-"*60)
    print("Testing edge cases...")
    print("-"*60)
    
    # Test 1: Empty embeddings
    is_valid, reason, _ = validate_embeddings([], [])
    assert not is_valid, "Empty embeddings should fail validation"
    print(f"✓ Empty embeddings rejected: {reason}")
    
    # Test 2: Mismatched counts
    is_valid, reason, _ = validate_embeddings(embeddings, texts[:2])
    assert not is_valid, "Mismatched counts should fail validation"
    print(f"✓ Mismatched counts rejected: {reason}")
    
    # Test 3: NaN values
    bad_embedding = embeddings.copy()
    bad_embedding[0] = [np.nan] * EXPECTED_EMBEDDING_DIM
    is_valid, reason, _ = validate_embeddings(bad_embedding, texts)
    assert not is_valid, "NaN values should fail validation"
    print(f"✓ NaN values rejected: {reason}")
    
    # Test 4: Inf values
    bad_embedding = embeddings.copy()
    bad_embedding[0] = [np.inf] * EXPECTED_EMBEDDING_DIM
    is_valid, reason, _ = validate_embeddings(bad_embedding, texts)
    assert not is_valid, "Inf values should fail validation"
    print(f"✓ Inf values rejected: {reason}")
    
    print("\n✅ All validation edge cases handled correctly")


def test_embedding_statistics(embeddings, texts):
    """Test the log_embedding_statistics function."""
    print("\n" + "="*60)
    print("TEST 4: Embedding Statistics")
    print("="*60)
    
    stats = log_embedding_statistics(embeddings, texts)
    
    # Verify statistics
    assert stats["total_embeddings"] == len(embeddings), "Should count all embeddings"
    assert stats["embedding_dimension"] == EXPECTED_EMBEDDING_DIM, \
        f"Should have dimension {EXPECTED_EMBEDDING_DIM}"
    assert stats["validation_passed"], "Valid embeddings should pass validation"
    
    # Check normalization (should be close to 1.0)
    assert 0.99 <= stats["avg_norm"] <= 1.01, "Embeddings should be normalized"
    
    print("\n✓ Statistics calculated correctly")
    print(f"✓ Average norm: {stats['avg_norm']:.4f} (expected ~1.0)")
    print(f"✓ Validation passed: {stats['validation_passed']}")
    
    return stats


def test_embedding_similarity():
    """Test that similar texts have similar embeddings."""
    print("\n" + "="*60)
    print("TEST 5: Embedding Similarity")
    print("="*60)
    
    model = get_embedding_model()
    
    # Similar texts
    text1 = "Machine learning is a type of artificial intelligence."
    text2 = "AI includes machine learning as one of its subfields."
    
    # Dissimilar text
    text3 = "The weather is sunny today."
    
    # Generate embeddings
    emb1 = np.array(model.embed_query(text1))
    emb2 = np.array(model.embed_query(text2))
    emb3 = np.array(model.embed_query(text3))
    
    # Calculate cosine similarities
    sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    print(f"\nSimilarity between similar texts: {sim_12:.4f}")
    print(f"Similarity between dissimilar texts: {sim_13:.4f}")
    
    # Similar texts should have higher similarity
    assert sim_12 > sim_13, "Similar texts should have higher similarity score"
    assert sim_12 > 0.5, "Similar texts should have positive correlation"
    
    print("\n✓ Similar texts have higher similarity scores")
    print("✓ Embeddings capture semantic meaning")


def main():
    """Run all tests."""
    print("\n" + "🧪 EMBEDDING VALIDATION TEST SUITE 🧪")
    
    try:
        # Test 1: Model loading
        model = test_embedding_model_loading()
        
        # Test 2: Embedding generation
        embeddings, texts = test_embedding_generation(model)
        
        # Test 3: Validation function
        test_validate_embeddings_function(embeddings, texts)
        
        # Test 4: Statistics
        stats = test_embedding_statistics(embeddings, texts)
        
        # Test 5: Similarity
        test_embedding_similarity()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nPhase 3 enhancements verified:")
        print("  ✓ Embedding model loading")
        print("  ✓ Embedding generation")
        print("  ✓ Embedding validation")
        print("  ✓ Embedding statistics")
        print(f"  ✓ Dimension verification ({EXPECTED_EMBEDDING_DIM})")
        print("  ✓ Normalization verification")
        print("  ✓ Semantic similarity")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
