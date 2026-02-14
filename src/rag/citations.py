"""
Citation Extraction and Validation

This module handles:
- Extracting citations from LLM responses
- Validating citations against retrieved documents
- Formatting source lists for display
- RAGResponse dataclass for structured output
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RAGResponse:
    """
    Structured response from RAG pipeline.
    
    Attributes:
        answer: LLM-generated answer
        sources: List of retrieved documents used as context
        citations: List of citation IDs found in answer
        confidence: Maximum similarity score from retrieval
        has_answer: True if answer was found (not fallback)
        metadata: Additional information (retrieval time, etc.)
    """
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    has_answer: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Answer: {self.answer}",
            f"Confidence: {self.confidence:.2f}",
            f"Has Answer: {self.has_answer}",
        ]
        
        if self.citations:
            lines.append(f"Citations: {', '.join(self.citations)}")
        
        if self.sources:
            lines.append(f"\nSources ({len(self.sources)}):")
            for idx, source in enumerate(self.sources):
                text_preview = source.get('text', source.get('page_content', ''))[:100]
                score = source.get('score', 'N/A')
                lines.append(f"  [{idx+1}] {text_preview}... (score: {score})")
        
        return "\n".join(lines)


def extract_citations(text: str) -> List[str]:
    """
    Extract citation IDs from text using pattern matching.
    
    Looks for patterns like [doc_001], [doc_123], etc.
    
    Args:
        text: Text containing potential citations
        
    Returns:
        List of unique citation IDs found
    """
    # Pattern: [doc_XXX] where XXX is 3 digits
    pattern = r'\[doc_\d{3}\]'
    matches = re.findall(pattern, text)
    
    # Remove brackets and return unique citations
    citations = list(set([match[1:-1] for match in matches]))  # Remove [ and ]
    return sorted(citations)  # Sort for consistency


def validate_citations(citations: List[str], valid_doc_ids: List[str]) -> Dict[str, Any]:
    """
    Validate that citations reference actual retrieved documents.
    
    Args:
        citations: List of citation IDs from answer
        valid_doc_ids: List of valid document IDs from retrieval
        
    Returns:
        Dictionary with validation results:
            - valid: List of valid citations
            - invalid: List of invalid citations
            - coverage: Percentage of sources cited (0-1)
            - all_valid: Boolean indicating if all citations are valid
    """
    valid = []
    invalid = []
    
    for citation in citations:
        if citation in valid_doc_ids:
            valid.append(citation)
        else:
            invalid.append(citation)
    
    # Calculate citation coverage (what % of sources were cited)
    coverage = len(set(valid)) / len(valid_doc_ids) if valid_doc_ids else 0.0
    
    return {
        'valid': valid,
        'invalid': invalid,
        'coverage': coverage,
        'all_valid': len(invalid) == 0,
        'num_citations': len(citations),
        'num_sources': len(valid_doc_ids)
    }


def format_sources(doc_ids: List[str], documents: List[Dict[str, Any]]) -> str:
    """
    Format source list for display to user.
    
    Args:
        doc_ids: List of document IDs
        documents: List of document dictionaries
        
    Returns:
        Formatted string listing all sources
    """
    if not documents:
        return "No sources available."
    
    lines = ["Sources:"]
    for idx, (doc_id, doc) in enumerate(zip(doc_ids, documents)):
        text = doc.get('text', doc.get('page_content', ''))
        text_preview = text[:100].replace('\n', ' ') + "..." if len(text) > 100 else text
        
        score = doc.get('score', 'N/A')
        if isinstance(score, float):
            score = f"{score:.4f}"
        
        lines.append(f"  - [{doc_id}] {text_preview} (similarity: {score})")
    
    return "\n".join(lines)


def check_fallback_response(answer: str) -> bool:
    """
    Check if answer is a fallback "I don't know" response.
    
    Args:
        answer: LLM response text
        
    Returns:
        True if answer is a fallback response
    """
    fallback_phrases = [
        "i don't have enough information",
        "i don't know",
        "cannot answer",
        "not enough information",
        "insufficient information",
        "no information",
    ]
    
    answer_lower = answer.lower().strip()
    return any(phrase in answer_lower for phrase in fallback_phrases)


def create_rag_response(
    answer: str,
    sources: List[Dict[str, Any]],
    doc_ids: List[str],
    max_score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> RAGResponse:
    """
    Create a structured RAGResponse from pipeline outputs.
    
    Args:
        answer: LLM-generated answer
        sources: Retrieved documents
        doc_ids: Document IDs used in context
        max_score: Maximum similarity score from retrieval
        metadata: Additional metadata
        
    Returns:
        RAGResponse object
    """
    # Extract citations from answer
    citations = extract_citations(answer)
    
    # Validate citations
    validation = validate_citations(citations, doc_ids)
    
    # Check if this is a fallback response
    has_answer = not check_fallback_response(answer)
    
    # Prepare metadata
    response_metadata = metadata or {}
    response_metadata['citation_validation'] = validation
    
    return RAGResponse(
        answer=answer,
        sources=sources,
        citations=citations,
        confidence=max_score,
        has_answer=has_answer,
        metadata=response_metadata
    )
