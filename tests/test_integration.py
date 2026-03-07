"""
Simple Integration Test - No Ollama Required

Tests the RAG pipeline components without requiring Ollama to be running.
Uses mocked LLM to verify the complete flow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from unittest.mock import Mock
from src.rag import RAGChain, RAGResponse
from src.rag.citations import extract_citations, validate_citations


def test_basic_query_flow():
    """Test Stage 2: Basic query flow with mocked components."""
    print("\n" + "="*80)
    print("TEST 1: Basic Query Flow")
    print("="*80)
    
    # Create mock retriever
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Machine learning is a subset of artificial intelligence that focuses on data."
    mock_doc.metadata = {'score': 0.85, 'source': 'ml_doc.txt'}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    # Create mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Machine learning is a subset of artificial intelligence [doc_000]."
    mock_llm.invoke.return_value = mock_response
    
    # Create RAG chain
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Execute query
    response = chain.query("What is machine learning?")
    
    # Verify
    print(f"✓ Query executed successfully")
    print(f"✓ Answer: {response.answer}")
    print(f"✓ Has answer: {response.has_answer}")
    print(f"✓ Confidence: {response.confidence}")
    print(f"✓ Citations: {response.citations}")
    
    assert response.has_answer is True, "Should have answer"
    assert response.confidence == 0.85, "Confidence should be 0.85"
    assert mock_retriever.retrieve.called, "Retriever should be called"
    assert mock_llm.invoke.called, "LLM should be called"
    
    print("\n✅ PASSED: Basic query flow works correctly\n")
    return True


def test_citation_extraction_accuracy():
    """Test Stage 5: Citation extraction accuracy."""
    print("="*80)
    print("TEST 2: Citation Extraction Accuracy")
    print("="*80)
    
    test_cases = [
        ("Answer with one citation [doc_000].", ["doc_000"]),
        ("Multiple citations [doc_000] and [doc_001].", ["doc_000", "doc_001"]),
        ("No citations at all.", []),
        ("Invalid format [doc] and [document_1].", []),
        ("Mixed [doc_000] and [doc] and [doc_001].", ["doc_000", "doc_001"]),
        ("Duplicate [doc_000] and [doc_000] again.", ["doc_000"]),
    ]
    
    passed = 0
    for text, expected in test_cases:
        result = extract_citations(text)
        if result == expected:
            print(f"✓ '{text[:50]}...' → {result}")
            passed += 1
        else:
            print(f"✗ '{text[:50]}...' → Expected {expected}, got {result}")
    
    accuracy = passed / len(test_cases) * 100
    print(f"\n✓ Citation extraction accuracy: {accuracy:.1f}% ({passed}/{len(test_cases)})")
    
    assert accuracy == 100, f"Expected 100% accuracy, got {accuracy}%"
    print("✅ PASSED: Citation extraction is 100% accurate\n")
    return True


def test_citation_validation_accuracy():
    """Test Stage 5: Citation validation accuracy."""
    print("="*80)
    print("TEST 3: Citation Validation Accuracy")
    print("="*80)
    
    # Test valid citations
    citations = ["doc_000", "doc_001"]
    valid_ids = ["doc_000", "doc_001", "doc_002"]
    result = validate_citations(citations, valid_ids)
    
    print(f"✓ Valid citations: {result['valid']}")
    print(f"✓ Invalid citations: {result['invalid']}")
    print(f"✓ All valid: {result['all_valid']}")
    print(f"✓ Coverage: {result['coverage']:.1%}")
    
    assert result['all_valid'] is True, "All citations should be valid"
    assert result['coverage'] == 2/3, "Coverage should be 66.7%"
    
    # Test invalid citations
    citations = ["doc_000", "doc_999"]
    result = validate_citations(citations, valid_ids)
    
    print(f"\n✓ Testing invalid citation detection:")
    print(f"  Valid: {result['valid']}")
    print(f"  Invalid: {result['invalid']}")
    
    assert "doc_999" in result['invalid'], "Should detect invalid citation"
    assert result['all_valid'] is False, "Should flag invalid citations"
    
    print("\n✅ PASSED: Citation validation works correctly\n")
    return True


def test_edge_case_low_confidence():
    """Test Stage 5: Edge case - low confidence."""
    print("="*80)
    print("TEST 4: Edge Case - Low Confidence")
    print("="*80)
    
    # Create mock retriever with low score
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Irrelevant text"
    mock_doc.metadata = {'score': 0.3}  # Below threshold
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("What is quantum computing?")
    
    print(f"✓ Low confidence score: {response.confidence}")
    print(f"✓ Has answer: {response.has_answer}")
    print(f"✓ Fallback reason: {response.metadata.get('fallback_reason')}")
    print(f"✓ Answer: {response.answer[:50]}...")
    
    assert response.has_answer is False, "Should not have answer"
    assert "don't have enough information" in response.answer.lower(), "Should return fallback"
    assert not mock_llm.invoke.called, "LLM should not be called"
    
    print("\n✅ PASSED: Low confidence edge case handled correctly\n")
    return True


def test_edge_case_no_context():
    """Test Stage 5: Edge case - no context."""
    print("="*80)
    print("TEST 5: Edge Case - No Context")
    print("="*80)
    
    # Create mock retriever with empty results
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = []
    
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Completely unrelated question")
    
    print(f"✓ No documents retrieved")
    print(f"✓ Has answer: {response.has_answer}")
    print(f"✓ Confidence: {response.confidence}")
    print(f"✓ Fallback reason: {response.metadata.get('fallback_reason')}")
    
    assert response.has_answer is False, "Should not have answer"
    assert response.confidence == 0.0, "Confidence should be 0"
    assert len(response.sources) == 0, "Should have no sources"
    assert not mock_llm.invoke.called, "LLM should not be called"
    
    print("\n✅ PASSED: No context edge case handled correctly\n")
    return True


def test_pipeline_performance():
    """Test Stage 5: Validate pipeline performance."""
    print("="*80)
    print("TEST 6: Pipeline Performance")
    print("="*80)
    
    import time
    
    # Create mock components
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Test document"
    mock_doc.metadata = {'score': 0.8}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Test answer [doc_000]."
    mock_llm.invoke.return_value = mock_response
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Run multiple queries and measure time
    num_queries = 10
    start_time = time.time()
    
    for i in range(num_queries):
        response = chain.query(f"Test question {i}")
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_queries
    
    print(f"✓ Total queries: {num_queries}")
    print(f"✓ Total time: {elapsed_time:.3f}s")
    print(f"✓ Average time per query: {avg_time:.3f}s")
    print(f"✓ Queries per second: {num_queries/elapsed_time:.1f}")
    
    # Performance assertions (with mocked LLM, should be very fast)
    assert avg_time < 1.0, f"Average query time should be < 1s, got {avg_time:.3f}s"
    
    print("\n✅ PASSED: Pipeline performance is acceptable\n")
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("RAG PIPELINE INTEGRATION TESTS")
    print("="*80)
    print("\nRunning tests without Ollama (using mocks)...\n")
    
    tests = [
        ("Basic Query Flow", test_basic_query_flow),
        ("Citation Extraction Accuracy", test_citation_extraction_accuracy),
        ("Citation Validation Accuracy", test_citation_validation_accuracy),
        ("Edge Case: Low Confidence", test_edge_case_low_confidence),
        ("Edge Case: No Context", test_edge_case_no_context),
        ("Pipeline Performance", test_pipeline_performance),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {name}")
            print(f"   Error: {e}\n")
            failed += 1
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    print("="*80)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! RAG pipeline is working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.\n")
        return 1


if __name__ == "__main__":
    exit(main())
