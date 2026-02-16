"""
FastAPI Backend for RAG System - Phase 5

This backend provides:
- /query endpoint for question answering
- Returns answer, sources, and confidence
- CORS support for frontend integration
- Error handling and validation
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import FAISSVectorStore
from src.retrieval import SemanticRetriever
from src.llm.model import get_llm
from src.rag import RAGChain, RAGResponse
from src.utils.persistence import load_vector_store
from config.settings import (
    DB_DIR,
    MIN_SIMILARITY_THRESHOLD,
    MAX_CONTEXT_DOCS,
    RETRIEVAL_K
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    k: Optional[int] = Field(None, ge=1, le=10, description="Number of documents to retrieve (optional)")


class Source(BaseModel):
    """Source document model."""
    text: str = Field(..., description="Source text content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Retrieved source documents")
    confidence: float = Field(..., description="Confidence score (0-1)")
    has_answer: bool = Field(..., description="Whether answer was found in sources")
    citations: List[str] = Field(default_factory=list, description="Extracted citations")
    query_time: float = Field(..., description="Query processing time in seconds")


# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval Augmented Generation API for question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for RAG components
rag_chain: Optional[RAGChain] = None
initialization_error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global rag_chain, initialization_error
    
    try:
        print("Initializing RAG system...")
        
        # Load vector store
        print("Loading vector store...")
        vector_store = load_vector_store(DB_DIR, "rag_index")
        print(f"Vector store loaded successfully")
        
        # Initialize retriever
        print("Initializing retriever...")
        retriever = SemanticRetriever(
            vector_store=vector_store,
            default_k=RETRIEVAL_K
        )
        
        # Initialize LLM
        print("Initializing Ollama LLM...")
        llm = get_llm()
        print("LLM initialized successfully")
        
        # Create RAG chain
        print("Creating RAG chain...")
        rag_chain = RAGChain(
            retriever=retriever,
            llm=llm,
            min_similarity=MIN_SIMILARITY_THRESHOLD,
            max_docs=MAX_CONTEXT_DOCS
        )
        
        print("RAG system initialized successfully!")
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {str(e)}"
        print(f"ERROR: {error_msg}")
        initialization_error = error_msg


@app.get("/")
async def root():
    """Root endpoint - health check."""
    if initialization_error:
        return {
            "status": "error",
            "message": initialization_error,
            "instructions": [
                "Ensure vector store exists (run pipeline_demo.py)",
                "Ensure Ollama is running (ollama serve)",
                "Ensure tinyllama model is available (ollama pull tinyllama)"
            ]
        }
    
    if rag_chain is None:
        return {
            "status": "initializing",
            "message": "RAG system is still initializing..."
        }
    
    return {
        "status": "ready",
        "message": "RAG system is ready",
        "endpoints": {
            "/query": "POST - Submit a question",
            "/health": "GET - Check system health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if initialization_error:
        raise HTTPException(status_code=503, detail=initialization_error)
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG system is initializing")
    
    return {
        "status": "healthy",
        "components": {
            "vector_store": "ready",
            "retriever": "ready",
            "llm": "ready",
            "rag_chain": "ready"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query endpoint - Answer questions using RAG pipeline.
    
    Args:
        request: QueryRequest with question and optional k parameter
        
    Returns:
        QueryResponse with answer, sources, confidence, and citations
    """
    # Check if system is initialized
    if initialization_error:
        raise HTTPException(
            status_code=503,
            detail=f"RAG system initialization failed: {initialization_error}"
        )
    
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing"
        )
    
    try:
        # Execute query
        start_time = time.time()
        rag_response: RAGResponse = rag_chain.query(request.question)
        query_time = time.time() - start_time
        
        # Debug: Check what sources actually are
        print(f"DEBUG: rag_response.sources type: {type(rag_response.sources)}")
        if rag_response.sources:
            print(f"DEBUG: First source type: {type(rag_response.sources[0])}")
            print(f"DEBUG: First source: {rag_response.sources[0]}")
        
        # Format sources
        sources = []
        for source in rag_response.sources:
            # Handle both dict and object types
            if isinstance(source, dict):
                text = source.get('text', '')
                score = source.get('score', 0.0)
                metadata = source.get('metadata', {})
            else:
                # Handle object with attributes - try text first, then page_content
                text = getattr(source, 'text', None) or getattr(source, 'page_content', '')
                score = getattr(source, 'score', 0.0)
                metadata = getattr(source, 'metadata', {})
            
            sources.append(Source(
                text=text,
                score=score,
                metadata=metadata
            ))
        
        # Build response
        response = QueryResponse(
            answer=rag_response.answer,
            sources=sources,
            confidence=rag_response.confidence,
            has_answer=rag_response.has_answer,
            citations=rag_response.citations,
            query_time=query_time
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting RAG Backend Server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
