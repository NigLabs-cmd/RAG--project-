"""
RAG Prompt Templates and Context Assembly

This module provides strict prompt engineering for context-only RAG:
- System prompts that enforce grounded answering
- Context formatting with document IDs for citation
- Prompt assembly with citation rules
"""

from typing import List, Dict, Any
import hashlib


# Strict RAG system prompt
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

RULES:
1. Answer ONLY using information from the context below
2. Cite sources using [doc_XXX] format after each claim or fact
3. If the context doesn't contain enough information to answer the question, respond EXACTLY: "I don't have enough information to answer this question."
4. Do not use any external knowledge or make assumptions
5. Be concise and factual

Your answers must be grounded in the context and include citations."""


def generate_doc_id(text: str, index: int) -> str:
    """
    Generate a unique document ID for citation tracking.
    
    Args:
        text: Document text content
        index: Document index in retrieval results
        
    Returns:
        Document ID in format 'doc_XXX'
    """
    # Use index-based ID for simplicity and consistency
    return f"doc_{index:03d}"


def format_context(documents: List[Dict[str, Any]], include_metadata: bool = False) -> tuple[str, List[str]]:
    """
    Format retrieved documents into structured context with IDs.
    
    Args:
        documents: List of retrieved documents with 'text' and optional 'metadata'
        include_metadata: Whether to include metadata in context
        
    Returns:
        Tuple of (formatted_context, list_of_doc_ids)
    """
    if not documents:
        return "", []
    
    context_parts = ["CONTEXT:"]
    doc_ids = []
    
    for idx, doc in enumerate(documents):
        doc_id = generate_doc_id(doc.get('text', ''), idx)
        doc_ids.append(doc_id)
        
        # Format document header
        context_parts.append(f"\n[Document {idx + 1} - ID: {doc_id}]")
        
        # Add metadata if requested
        if include_metadata and 'metadata' in doc:
            metadata = doc['metadata']
            if metadata:
                meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                context_parts.append(f"Metadata: {meta_str}")
        
        # Add document text
        text = doc.get('text', doc.get('page_content', ''))
        context_parts.append(text.strip())
    
    return "\n".join(context_parts), doc_ids


def create_rag_prompt(question: str, documents: List[Dict[str, Any]], include_metadata: bool = False) -> tuple[str, List[str]]:
    """
    Create complete RAG prompt with context and question.
    
    Args:
        question: User's question
        documents: Retrieved documents
        include_metadata: Whether to include document metadata
        
    Returns:
        Tuple of (complete_prompt, list_of_doc_ids)
    """
    context, doc_ids = format_context(documents, include_metadata)
    
    if not context:
        # No context available
        prompt = f"""{RAG_SYSTEM_PROMPT}

CONTEXT:
[No relevant documents found]

QUESTION: {question}

ANSWER:"""
        return prompt, []
    
    prompt = f"""{RAG_SYSTEM_PROMPT}

{context}

QUESTION: {question}

ANSWER:"""
    
    return prompt, doc_ids


def create_fallback_response() -> str:
    """
    Create standard fallback response for low-confidence retrievals.
    
    Returns:
        Fallback message
    """
    return "I don't have enough information to answer this question."


# Alternative prompt templates for different use cases

CONVERSATIONAL_SYSTEM_PROMPT = """You are a friendly assistant that helps users find information from documents.

GUIDELINES:
- Answer based on the provided context
- Use a conversational, helpful tone
- Cite sources using [doc_XXX] format
- If unsure, say so politely

Be helpful and accurate."""


def create_conversational_prompt(question: str, documents: List[Dict[str, Any]]) -> tuple[str, List[str]]:
    """
    Create a more conversational RAG prompt (less strict).
    
    Args:
        question: User's question
        documents: Retrieved documents
        
    Returns:
        Tuple of (complete_prompt, list_of_doc_ids)
    """
    context, doc_ids = format_context(documents, include_metadata=False)
    
    prompt = f"""{CONVERSATIONAL_SYSTEM_PROMPT}

{context}

USER QUESTION: {question}

ASSISTANT:"""
    
    return prompt, doc_ids
