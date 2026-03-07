"""
RAG Module Test Suite

Tests for:
- Prompt formatting and context assembly
- Citation extraction and validation
- RAG chain orchestration
- Edge cases and fallback handling
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.rag.prompts import (
    generate_doc_id,
    format_context,
    create_rag_prompt,
    create_fallback_response,
    RAG_SYSTEM_PROMPT
)
from src.rag.citations import (
    extract_citations,
    validate_citations,
    format_sources,
    check_fallback_response,
    create_rag_response,
    RAGResponse
)
from src.rag.chain import RAGChain


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_DOCS = [
    {
        'text': 'Machine learning is a subset of artificial intelligence.',
        'metadata': {'source': 'ml_doc.txt'},
        'score': 0.85
    },
    {
        'text': 'Python is a popular programming language for data science.',
        'metadata': {'source': 'python_doc.txt'},
        'score': 0.72
    },
    {
        'text': 'Neural networks are inspired by biological neurons.',
        'metadata': {'source': 'nn_doc.txt'},
        'score': 0.68
    }
]


# ============================================================================
# Prompt Tests
# ============================================================================

def test_generate_doc_id():
    """Test document ID generation."""
    doc_id = generate_doc_id("some text", 0)
    assert doc_id == "doc_000"
    
    doc_id = generate_doc_id("other text", 5)
    assert doc_id == "doc_005"
    
    doc_id = generate_doc_id("", 123)
    assert doc_id == "doc_123"


def test_format_context_basic():
    """Test basic context formatting."""
    context, doc_ids = format_context(SAMPLE_DOCS[:2])
    
    assert "CONTEXT:" in context
    assert "[Document 1 - ID: doc_000]" in context
    assert "[Document 2 - ID: doc_001]" in context
    assert "Machine learning" in context
    assert "Python" in context
    
    assert len(doc_ids) == 2
    assert doc_ids[0] == "doc_000"
    assert doc_ids[1] == "doc_001"


def test_format_context_with_metadata():
    """Test context formatting with metadata."""
    context, doc_ids = format_context(SAMPLE_DOCS[:1], include_metadata=True)
    
    assert "Metadata:" in context
    assert "source: ml_doc.txt" in context


def test_format_context_empty():
    """Test context formatting with empty documents."""
    context, doc_ids = format_context([])
    
    assert context == ""
    assert doc_ids == []


def test_create_rag_prompt():
    """Test RAG prompt creation."""
    question = "What is machine learning?"
    prompt, doc_ids = create_rag_prompt(question, SAMPLE_DOCS[:2])
    
    assert RAG_SYSTEM_PROMPT in prompt
    assert "CONTEXT:" in prompt
    assert "QUESTION: What is machine learning?" in prompt
    assert "ANSWER:" in prompt
    assert "[Document 1 - ID: doc_000]" in prompt
    
    assert len(doc_ids) == 2


def test_create_rag_prompt_no_docs():
    """Test RAG prompt with no documents."""
    question = "What is quantum computing?"
    prompt, doc_ids = create_rag_prompt(question, [])
    
    assert "[No relevant documents found]" in prompt
    assert "QUESTION: What is quantum computing?" in prompt
    assert doc_ids == []


def test_create_fallback_response():
    """Test fallback response creation."""
    response = create_fallback_response()
    assert "don't have enough information" in response.lower()


# ============================================================================
# Citation Tests
# ============================================================================

def test_extract_citations_valid():
    """Test citation extraction with valid patterns."""
    text = "Machine learning is AI [doc_000]. Python is popular [doc_001]."
    citations = extract_citations(text)
    
    assert len(citations) == 2
    assert "doc_000" in citations
    assert "doc_001" in citations


def test_extract_citations_multiple_same():
    """Test citation extraction with duplicate citations."""
    text = "ML is AI [doc_000]. AI includes ML [doc_000]."
    citations = extract_citations(text)
    
    # Should deduplicate
    assert len(citations) == 1
    assert "doc_000" in citations


def test_extract_citations_invalid_patterns():
    """Test that invalid citation patterns are not extracted."""
    text = "This has [doc] and [document_1] and [doc_1] but not valid format."
    citations = extract_citations(text)
    
    # Only [doc_XXX] with 3 digits should match
    assert len(citations) == 0


def test_extract_citations_none():
    """Test citation extraction with no citations."""
    text = "This text has no citations at all."
    citations = extract_citations(text)
    
    assert len(citations) == 0


def test_validate_citations_all_valid():
    """Test citation validation with all valid citations."""
    citations = ["doc_000", "doc_001"]
    valid_ids = ["doc_000", "doc_001", "doc_002"]
    
    result = validate_citations(citations, valid_ids)
    
    assert result['all_valid'] is True
    assert len(result['valid']) == 2
    assert len(result['invalid']) == 0
    assert result['coverage'] == 2/3  # 2 out of 3 sources cited


def test_validate_citations_some_invalid():
    """Test citation validation with some invalid citations."""
    citations = ["doc_000", "doc_999"]  # doc_999 doesn't exist
    valid_ids = ["doc_000", "doc_001"]
    
    result = validate_citations(citations, valid_ids)
    
    assert result['all_valid'] is False
    assert "doc_000" in result['valid']
    assert "doc_999" in result['invalid']
    assert result['coverage'] == 0.5  # 1 out of 2 sources cited


def test_format_sources():
    """Test source formatting for display."""
    doc_ids = ["doc_000", "doc_001"]
    sources_text = format_sources(doc_ids, SAMPLE_DOCS[:2])
    
    assert "Sources:" in sources_text
    assert "[doc_000]" in sources_text
    assert "[doc_001]" in sources_text
    assert "similarity:" in sources_text


def test_check_fallback_response_positive():
    """Test fallback response detection - positive cases."""
    assert check_fallback_response("I don't have enough information to answer this question.")
    assert check_fallback_response("I don't know the answer.")
    assert check_fallback_response("Cannot answer based on the context.")
    assert check_fallback_response("Not enough information available.")


def test_check_fallback_response_negative():
    """Test fallback response detection - negative cases."""
    assert not check_fallback_response("Machine learning is a subset of AI [doc_000].")
    assert not check_fallback_response("Python is used for data science.")


def test_rag_response_dataclass():
    """Test RAGResponse dataclass creation."""
    response = RAGResponse(
        answer="Test answer [doc_000]",
        sources=SAMPLE_DOCS[:1],
        citations=["doc_000"],
        confidence=0.85,
        has_answer=True,
        metadata={'query_time': 1.5}
    )
    
    assert response.answer == "Test answer [doc_000]"
    assert len(response.sources) == 1
    assert response.citations == ["doc_000"]
    assert response.confidence == 0.85
    assert response.has_answer is True
    assert response.metadata['query_time'] == 1.5


def test_create_rag_response():
    """Test RAGResponse creation with citation extraction."""
    answer = "Machine learning is AI [doc_000]. Python is popular [doc_001]."
    doc_ids = ["doc_000", "doc_001"]
    
    response = create_rag_response(
        answer=answer,
        sources=SAMPLE_DOCS[:2],
        doc_ids=doc_ids,
        max_score=0.85
    )
    
    assert response.answer == answer
    assert len(response.citations) == 2
    assert response.confidence == 0.85
    assert response.has_answer is True
    
    # Check citation validation in metadata
    validation = response.metadata['citation_validation']
    assert validation['all_valid'] is True


# ============================================================================
# RAG Chain Tests
# ============================================================================

def test_rag_chain_initialization():
    """Test RAG chain initialization."""
    mock_retriever = Mock()
    mock_llm = Mock()
    
    chain = RAGChain(
        retriever=mock_retriever,
        llm=mock_llm,
        min_similarity=0.5,
        max_docs=3
    )
    
    assert chain.retriever == mock_retriever
    assert chain.llm == mock_llm
    assert chain.min_similarity == 0.5
    assert chain.max_docs == 3


def test_rag_chain_check_confidence_pass():
    """Test confidence check with high scores."""
    mock_retriever = Mock()
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # High confidence docs
    docs = [{'score': 0.85}, {'score': 0.72}]
    assert chain._check_confidence(docs) is True


def test_rag_chain_check_confidence_fail():
    """Test confidence check with low scores."""
    mock_retriever = Mock()
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Low confidence docs
    docs = [{'score': 0.3}, {'score': 0.2}]
    assert chain._check_confidence(docs) is False


def test_rag_chain_check_confidence_empty():
    """Test confidence check with empty results."""
    mock_retriever = Mock()
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    assert chain._check_confidence([]) is False


def test_rag_chain_fallback_response():
    """Test fallback response creation."""
    mock_retriever = Mock()
    mock_llm = Mock()
    
    chain = RAGChain(mock_retriever, mock_llm)
    
    response = chain._create_fallback_response([], 0.0)
    
    assert response.has_answer is False
    assert "don't have enough information" in response.answer.lower()
    assert response.confidence == 0.0


# ============================================================================
# Integration Tests (with mocks)
# ============================================================================

def test_rag_chain_query_success():
    """Test successful RAG chain query."""
    # Create mock retriever
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Machine learning is a subset of AI."
    mock_doc.metadata = {'score': 0.85, 'source': 'ml.txt'}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    # Create mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Machine learning is a subset of artificial intelligence [doc_000]."
    mock_llm.invoke.return_value = mock_response
    
    # Create chain
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Execute query
    response = chain.query("What is machine learning?")
    
    # Verify
    assert response.has_answer is True
    assert response.confidence >= 0.5
    assert len(response.citations) > 0
    assert mock_retriever.retrieve.called
    assert mock_llm.invoke.called


def test_rag_chain_query_low_confidence():
    """Test RAG chain query with low confidence retrieval."""
    # Create mock retriever with low scores
    mock_retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Irrelevant text."
    mock_doc.metadata = {'score': 0.2, 'source': 'irrelevant.txt'}
    mock_retriever.retrieve.return_value = [mock_doc]
    
    # Create mock LLM (should not be called)
    mock_llm = Mock()
    
    # Create chain
    chain = RAGChain(mock_retriever, mock_llm, min_similarity=0.5)
    
    # Execute query
    response = chain.query("What is quantum computing?")
    
    # Verify fallback
    assert response.has_answer is False
    assert "don't have enough information" in response.answer.lower()
    assert not mock_llm.invoke.called  # LLM should not be invoked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
