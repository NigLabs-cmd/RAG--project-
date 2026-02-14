"""
RAG Module - Retrieval Augmented Generation

This module provides end-to-end RAG pipeline with:
- Strict prompt engineering for grounded answering
- Citation extraction and validation
- Confidence-based fallback handling
- Ollama LLM integration
"""

from .chain import RAGChain, create_rag_chain
from .citations import RAGResponse, extract_citations, validate_citations, format_sources
from .prompts import create_rag_prompt, format_context, RAG_SYSTEM_PROMPT

__all__ = [
    # Main classes
    "RAGChain",
    "RAGResponse",
    
    # Factory functions
    "create_rag_chain",
    
    # Prompt utilities
    "create_rag_prompt",
    "format_context",
    "RAG_SYSTEM_PROMPT",
    
    # Citation utilities
    "extract_citations",
    "validate_citations",
    "format_sources",
]
