"""
Test script for metadata enhancement functionality.
Tests chunk_index, character positions, timestamps, and metadata validation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.splitter import split_documents, validate_metadata
from datetime import datetime


def test_metadata_enhancement():
    """Test that all metadata fields are properly added."""
    print("\n" + "="*60)
    print("TEST: Metadata Enhancement")
    print("="*60)
    
    # Create sample documents
    sample_docs = [
        {
            "content": """
            Artificial Intelligence (AI) is transforming how we interact with technology.
            From virtual assistants to autonomous vehicles, AI systems are becoming 
            increasingly sophisticated and capable of handling complex tasks.
            
            Machine learning, a subset of AI, enables systems to learn from data without
            explicit programming. This has led to breakthroughs in image recognition,
            natural language processing, and predictive analytics.
            """,
            "metadata": {"source": "ai_overview.txt", "page": 1}
        },
        {
            "content": """
            Python has emerged as the leading language for AI and machine learning development.
            Its simple syntax and extensive library ecosystem make it ideal for rapid 
            prototyping and production deployment.
            """,
            "metadata": {"source": "python_ml.txt", "page": 1}
        }
    ]
    
    # Process documents
    print("\n📝 Processing documents...")
    chunks = split_documents(sample_docs)
    
    print(f"\n✓ Generated {len(chunks)} chunks")
    
    # Test 1: Chunk Index
    print("\n" + "-"*60)
    print("TEST 1: Chunk Index")
    print("-"*60)
    for i, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata, f"Chunk {i} missing chunk_index"
        assert chunk.metadata["chunk_index"] == i, f"Chunk {i} has wrong index"
        print(f"  Chunk {i}: chunk_index = {chunk.metadata['chunk_index']} ✓")
    print("✅ All chunks have correct indices")
    
    # Test 2: Character Positions
    print("\n" + "-"*60)
    print("TEST 2: Character Positions")
    print("-"*60)
    for i, chunk in enumerate(chunks):
        assert "char_start" in chunk.metadata, f"Chunk {i} missing char_start"
        assert "char_end" in chunk.metadata, f"Chunk {i} missing char_end"
        
        char_start = chunk.metadata["char_start"]
        char_end = chunk.metadata["char_end"]
        chunk_length = len(chunk.page_content)
        
        assert char_end - char_start == chunk_length, f"Chunk {i} position mismatch"
        print(f"  Chunk {i}: [{char_start:4d} - {char_end:4d}] ({chunk_length} chars) ✓")
    
    # Verify positions are sequential
    for i in range(len(chunks) - 1):
        current_end = chunks[i].metadata["char_end"]
        next_start = chunks[i+1].metadata["char_start"]
        # Note: positions may not be perfectly sequential due to overlap
        print(f"  Gap between chunk {i} and {i+1}: {next_start - current_end} chars")
    
    print("✅ All chunks have valid character positions")
    
    # Test 3: Processing Timestamp
    print("\n" + "-"*60)
    print("TEST 3: Processing Timestamp")
    print("-"*60)
    timestamps = set()
    for i, chunk in enumerate(chunks):
        assert "processed_at" in chunk.metadata, f"Chunk {i} missing processed_at"
        
        timestamp_str = chunk.metadata["processed_at"]
        timestamps.add(timestamp_str)
        
        # Verify it's a valid ISO format timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            print(f"  Chunk {i}: {timestamp_str} ✓")
        except ValueError:
            raise AssertionError(f"Chunk {i} has invalid timestamp format")
    
    # All chunks from same processing run should have same timestamp
    assert len(timestamps) == 1, "All chunks should have same processing timestamp"
    print(f"✅ All chunks have valid timestamp: {list(timestamps)[0]}")
    
    # Test 4: Quality Validation Metadata
    print("\n" + "-"*60)
    print("TEST 4: Quality Validation Metadata")
    print("-"*60)
    for i, chunk in enumerate(chunks):
        assert "quality_valid" in chunk.metadata, f"Chunk {i} missing quality_valid"
        
        is_valid = chunk.metadata["quality_valid"]
        quality_reason = chunk.metadata.get("quality_reason", "Valid")
        
        print(f"  Chunk {i}: quality_valid={is_valid}, reason='{quality_reason}' ✓")
    
    print("✅ All chunks have quality validation metadata")
    
    # Test 5: Metadata Validation
    print("\n" + "-"*60)
    print("TEST 5: Metadata Validation Function")
    print("-"*60)
    valid_count = 0
    invalid_count = 0
    
    for i, chunk in enumerate(chunks):
        is_valid, reason = validate_metadata(chunk.metadata)
        if is_valid:
            valid_count += 1
            print(f"  Chunk {i}: ✓ Valid")
        else:
            invalid_count += 1
            print(f"  Chunk {i}: ✗ Invalid - {reason}")
    
    print(f"\n  Valid metadata: {valid_count}/{len(chunks)}")
    print(f"  Invalid metadata: {invalid_count}/{len(chunks)}")
    
    assert valid_count == len(chunks), "All chunks should have valid metadata"
    print("✅ All chunks pass metadata validation")
    
    # Test 6: Original Metadata Preserved
    print("\n" + "-"*60)
    print("TEST 6: Original Metadata Preserved")
    print("-"*60)
    for i, chunk in enumerate(chunks):
        assert "source" in chunk.metadata, f"Chunk {i} missing source"
        assert "page" in chunk.metadata, f"Chunk {i} missing page"
        print(f"  Chunk {i}: source='{chunk.metadata['source']}', page={chunk.metadata['page']} ✓")
    
    print("✅ Original metadata preserved")
    
    # Summary
    print("\n" + "="*60)
    print("📊 METADATA SUMMARY")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    print(f"\nMetadata fields per chunk:")
    if chunks:
        for key in sorted(chunks[0].metadata.keys()):
            print(f"  - {key}")
    
    print("\n" + "="*60)
    print("🎉 ALL METADATA TESTS PASSED! 🎉")
    print("="*60)
    print("\nPhase 2 enhancements verified:")
    print("  ✓ Chunk indices")
    print("  ✓ Character positions (start/end)")
    print("  ✓ Processing timestamps")
    print("  ✓ Quality validation flags")
    print("  ✓ Metadata validation")
    print("  ✓ Original metadata preservation")


def main():
    """Run all tests."""
    try:
        test_metadata_enhancement()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
