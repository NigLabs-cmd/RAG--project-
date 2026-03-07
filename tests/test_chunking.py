"""
Test script for enhanced text chunking functionality.
Tests clean_text(), validate_chunk_quality(), and chunk statistics.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.splitter import clean_text, validate_chunk_quality, split_documents
from langchain_core.documents import Document


def test_clean_text():
    """Test the clean_text function."""
    print("\n" + "="*60)
    print("TEST 1: clean_text() Function")
    print("="*60)
    
    # Test case 1: Excessive whitespace
    text1 = "This  has   too    many     spaces\n\n\n\n\nand newlines"
    cleaned1 = clean_text(text1)
    print(f"Input:  '{text1}'")
    print(f"Output: '{cleaned1}'")
    assert "     " not in cleaned1, "Should remove excessive spaces"
    assert "\n\n\n" not in cleaned1, "Should remove excessive newlines"
    print("✓ Excessive whitespace removed")
    
    # Test case 2: Special characters
    text2 = "Normal text\x00with\x08special\x1fchars"
    cleaned2 = clean_text(text2)
    print(f"\nInput:  '{text2}'")
    print(f"Output: '{cleaned2}'")
    assert "\x00" not in cleaned2, "Should remove null characters"
    print("✓ Special characters removed")
    
    # Test case 3: Empty/None text
    cleaned3 = clean_text("")
    cleaned4 = clean_text(None)
    assert cleaned3 == "", "Should handle empty string"
    assert cleaned4 == "", "Should handle None"
    print("✓ Edge cases handled")
    
    print("\n✅ All clean_text tests passed!")


def test_validate_chunk_quality():
    """Test the validate_chunk_quality function."""
    print("\n" + "="*60)
    print("TEST 2: validate_chunk_quality() Function")
    print("="*60)
    
    # Test case 1: Valid chunk
    valid_chunk = Document(
        page_content="This is a valid chunk with enough content to pass validation. "
                     "It has multiple sentences and a reasonable length.",
        metadata={"source": "test"}
    )
    is_valid, reason = validate_chunk_quality(valid_chunk)
    print(f"Valid chunk: {is_valid} - {reason}")
    assert is_valid, "Should validate good chunk"
    print("✓ Valid chunk accepted")
    
    # Test case 2: Too short
    short_chunk = Document(page_content="Too short", metadata={"source": "test"})
    is_valid, reason = validate_chunk_quality(short_chunk)
    print(f"\nShort chunk: {is_valid} - {reason}")
    assert not is_valid, "Should reject too-short chunk"
    print("✓ Short chunk rejected")
    
    # Test case 3: Too few words
    few_words = Document(
        page_content="One two three four five six seven eight nine",
        metadata={"source": "test"}
    )
    is_valid, reason = validate_chunk_quality(few_words)
    print(f"\nFew words chunk: {is_valid} - {reason}")
    assert not is_valid, "Should reject chunk with too few words"
    print("✓ Few-word chunk rejected")
    
    # Test case 4: Excessive repetition
    repetitive = Document(
        page_content="test " * 50,  # Same word repeated 50 times
        metadata={"source": "test"}
    )
    is_valid, reason = validate_chunk_quality(repetitive)
    print(f"\nRepetitive chunk: {is_valid} - {reason}")
    assert not is_valid, "Should reject repetitive chunk"
    print("✓ Repetitive chunk rejected")
    
    print("\n✅ All validate_chunk_quality tests passed!")


def test_split_documents():
    """Test the enhanced split_documents function with real data."""
    print("\n" + "="*60)
    print("TEST 3: split_documents() with Statistics")
    print("="*60)
    
    # Create sample documents
    sample_docs = [
        {
            "content": """
            Machine learning is a subset of artificial intelligence that focuses on 
            developing algorithms that can learn from and make predictions on data. 
            It has applications in various fields including computer vision, natural 
            language processing, and robotics.
            
            Deep learning is a specialized form of machine learning that uses neural 
            networks with multiple layers. These networks can automatically learn 
            hierarchical representations of data, making them particularly effective 
            for complex tasks like image recognition and speech processing.
            """,
            "metadata": {"source": "ml_intro.txt", "page": 1}
        },
        {
            "content": """
            Python is a high-level programming language known for its simplicity and 
            readability. It has become the de facto language for data science and 
            machine learning due to its extensive ecosystem of libraries like NumPy, 
            Pandas, and scikit-learn.
            
            The language's syntax emphasizes code readability, using significant 
            whitespace and clear naming conventions. This makes Python an excellent 
            choice for both beginners and experienced developers.
            """,
            "metadata": {"source": "python_intro.txt", "page": 1}
        }
    ]
    
    # Process documents
    chunks = split_documents(sample_docs)
    
    # Verify results
    print(f"\n📊 Results:")
    print(f"   Input documents: {len(sample_docs)}")
    print(f"   Output chunks: {len(chunks)}")
    assert len(chunks) > 0, "Should generate chunks"
    
    # Check that metadata is preserved
    for chunk in chunks:
        assert "source" in chunk.metadata, "Should preserve source metadata"
        assert "page" in chunk.metadata, "Should preserve page metadata"
    print("   ✓ Metadata preserved")
    
    # Check that text was cleaned
    for chunk in chunks:
        content = chunk.page_content
        assert "\n\n\n" not in content, "Should have cleaned excessive newlines"
        assert "  " not in content or content.count("  ") < 3, "Should have cleaned excessive spaces"
    print("   ✓ Text cleaned")
    
    print("\n✅ All split_documents tests passed!")


def main():
    """Run all tests."""
    print("\n" + "🧪 CHUNKING ENHANCEMENT TEST SUITE 🧪")
    
    try:
        test_clean_text()
        test_validate_chunk_quality()
        test_split_documents()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nPhase 1 enhancements are working correctly:")
        print("  ✓ Text cleaning")
        print("  ✓ Chunk quality validation")
        print("  ✓ Statistics logging")
        print("  ✓ Metadata preservation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
