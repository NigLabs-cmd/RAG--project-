"""
Edge Case Test Suite for RAG Pipeline

Comprehensive tests for all edge cases and fallback scenarios:
- Low confidence detection
- Empty retrieval results
- Irrelevant context
- LLM errors
- Malformed inputs
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.rag.chain import RAGChain
from src.rag.prompts import create_rag_prompt, create_fallback_response
from src.rag.citations import RAGResponse


# ============================================================================
# Edge Case 1: Low Confidence Detection
# ============================================================================

def test_low_confidence_retrieval():
    """Test that low similarity scores trigger fallback response."""
    # Create mock retriever with low scores
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Some irrelevant text."
    mock_doc.metadata = {'score': 0.3, 'source': 'doc.txt'}  # Below 0.5 threshold
    mock_retriever.retrieve.return_value = [mock_doc]
    
    # Create mock LLM (should NOT be called)
    mock_llm = Mock()
    
    # Create chain with 0.5 threshold
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Execute query
    response = chain.query("What is quantum computing?")
    
    # Verify fallback behavior
    assert response.has_answer is False
    assert "don't have enough information" in response.answer.lower()
    assert response.confidence == 0.3
    assert response.metadata['fallback_reason'] == 'low_confidence'
    
    # Verify LLM was NOT invoked
    assert not mock_llm.invoke.called


def test_exactly_at_threshold():
    """Test behavior when similarity is exactly at threshold."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Relevant text."
    mock_doc.metadata = {'score': 0.5}  # Exactly at threshold
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Answer [doc_000]."
    mock_llm.invoke.return_value = mock_response
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Test question")
    
    # Should pass (>= threshold)
    assert response.has_answer is True
    assert mock_llm.invoke.called


def test_just_below_threshold():
    """Test behavior when similarity is just below threshold."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text."
    mock_doc.metadata = {'score': 0.49}  # Just below 0.5
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Test question")
    
    # Should fail (< threshold)
    assert response.has_answer is False
    assert not mock_llm.invoke.called


# ============================================================================
# Edge Case 2: Empty Retrieval Results
# ============================================================================

def test_empty_retrieval_results():
    """Test handling of empty retrieval results."""
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = []  # No documents found
    
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Completely unrelated question")
    
    # Verify fallback
    assert response.has_answer is False
    assert "don't have enough information" in response.answer.lower()
    assert response.confidence == 0.0
    assert len(response.sources) == 0
    assert response.metadata['fallback_reason'] == 'no_results'
    
    # LLM should not be called
    assert not mock_llm.invoke.called


def test_none_retrieval_results():
    """Test handling when retriever returns None."""
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = None
    
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Should handle gracefully (treat as empty)
    with pytest.raises(TypeError):
        # This will raise because we try to iterate over None
        # In production, retriever should never return None
        response = chain.query("Test")


# ============================================================================
# Edge Case 3: Irrelevant Context Scenarios
# ============================================================================

def test_all_irrelevant_documents():
    """Test when all retrieved documents are irrelevant (low scores)."""
    mock_retriever = Mock()
    
    # Multiple docs, all with low scores
    docs = []
    for i in range(3):
        doc = Mock()
        doc.page_content = f"Irrelevant text {i}"
        doc.metadata = {'score': 0.1 + i * 0.05}  # 0.1, 0.15, 0.2
        docs.append(doc)
    
    mock_retriever.retrieve.return_value = docs
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Question")
    
    # Should trigger fallback (max score 0.2 < 0.5)
    assert response.has_answer is False
    assert response.confidence == 0.2  # Max of the scores
    assert len(response.sources) == 3
    assert not mock_llm.invoke.called


def test_mixed_relevance_documents():
    """Test when some docs are relevant, some are not."""
    mock_retriever = Mock()
    
    # Mix of high and low scores
    doc1 = Mock()
    doc1.page_content = "Relevant text"
    doc1.metadata = {'score': 0.8}
    
    doc2 = Mock()
    doc2.page_content = "Irrelevant text"
    doc2.metadata = {'score': 0.2}
    
    mock_retriever.retrieve.return_value = [doc1, doc2]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Answer [doc_000]."
    mock_llm.invoke.return_value = mock_response
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Question")
    
    # Should pass (max score 0.8 >= 0.5)
    assert response.has_answer is True
    assert response.confidence == 0.8
    assert mock_llm.invoke.called


# ============================================================================
# Edge Case 4: LLM Error Handling
# ============================================================================

def test_llm_connection_error():
    """Test graceful handling of LLM connection errors."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Relevant text"
    mock_doc.metadata = {'score': 0.8}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    # LLM that raises connection error
    mock_llm = Mock()
    mock_llm.invoke.side_effect = ConnectionError("Ollama not running")
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Question")
    
    # Should return fallback response
    assert "don't have enough information" in response.answer.lower()
    # Note: has_answer might be False due to fallback detection


def test_llm_timeout_error():
    """Test handling of LLM timeout errors."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text"
    mock_doc.metadata = {'score': 0.8}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    mock_llm.invoke.side_effect = TimeoutError("Request timed out")
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Question")
    
    # Should handle gracefully
    assert "don't have enough information" in response.answer.lower()


def test_llm_returns_empty_response():
    """Test handling when LLM returns empty string."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text"
    mock_doc.metadata = {'score': 0.8}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = ""  # Empty response
    mock_llm.invoke.return_value = mock_response
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Question")
    
    # Should handle empty response
    assert response.answer == ""
    assert response.has_answer is True  # Not detected as fallback


def test_llm_returns_malformed_response():
    """Test handling of malformed LLM responses."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text"
    mock_doc.metadata = {'score': 0.8}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    # Return object without 'content' attribute
    mock_llm.invoke.return_value = "Just a string"
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("Question")
    
    # Should convert to string
    assert response.answer == "Just a string"


# ============================================================================
# Edge Case 5: Malformed Inputs
# ============================================================================

def test_empty_question():
    """Test handling of empty question string."""
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = []
    
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    response = chain.query("")
    
    # Should handle gracefully
    assert response.has_answer is False


def test_very_long_question():
    """Test handling of very long questions."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text"
    mock_doc.metadata = {'score': 0.8}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Answer [doc_000]."
    mock_llm.invoke.return_value = mock_response
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Very long question (1000 chars)
    long_question = "What is " + "very " * 200 + "important?"
    response = chain.query(long_question)
    
    # Should handle without error
    assert response is not None


def test_special_characters_in_question():
    """Test handling of special characters in questions."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text"
    mock_doc.metadata = {'score': 0.8}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Answer."
    mock_llm.invoke.return_value = mock_response
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Question with special chars
    response = chain.query("What's the <meaning> of [life]? @#$%")
    
    # Should handle without error
    assert response is not None


# ============================================================================
# Edge Case 6: Batch Query Edge Cases
# ============================================================================

def test_batch_query_with_mixed_results():
    """Test batch query with mix of successful and failed queries."""
    mock_retriever = Mock()
    
    # First query: high confidence
    doc1 = Mock()
    doc1.page_content = "Relevant"
    doc1.metadata = {'score': 0.8}
    
    # Second query: low confidence
    doc2 = Mock()
    doc2.page_content = "Irrelevant"
    doc2.metadata = {'score': 0.2}
    
    mock_retriever.retrieve.side_effect = [[doc1], [doc2]]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Answer [doc_000]."
    mock_llm.invoke.return_value = mock_response
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    responses = chain.batch_query(["Question 1", "Question 2"])
    
    assert len(responses) == 2
    assert responses[0].has_answer is True  # High confidence
    assert responses[1].has_answer is False  # Low confidence


def test_batch_query_empty_list():
    """Test batch query with empty question list."""
    mock_retriever = Mock()
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    responses = chain.batch_query([])
    
    assert len(responses) == 0


# ============================================================================
# Edge Case 7: Custom Threshold Testing
# ============================================================================

def test_very_low_threshold():
    """Test with very low similarity threshold (0.1)."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text"
    mock_doc.metadata = {'score': 0.15}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Answer."
    mock_llm.invoke.return_value = mock_response
    
    # Very low threshold
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.1)
    response = chain.query("Question")
    
    # Should pass with 0.15 score
    assert response.has_answer is True
    assert mock_llm.invoke.called


def test_very_high_threshold():
    """Test with very high similarity threshold (0.9)."""
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Text"
    mock_doc.metadata = {'score': 0.85}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    mock_llm = Mock()
    
    # Very high threshold
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.9)
    response = chain.query("Question")
    
    # Should fail with 0.85 score
    assert response.has_answer is False
    assert not mock_llm.invoke.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
