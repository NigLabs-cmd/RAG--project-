"""
FastAPI Backend for RAG System - Phase 5

This backend provides:
- /query endpoint for question answering
- /upload endpoint for PDF document ingestion
- /documents endpoint for listing uploaded documents
- Returns answer, sources, and confidence
- CORS support for frontend integration
- Error handling and validation
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import FAISSVectorStore
from src.retrieval import SemanticRetriever
from src.llm.model import get_llm
from src.rag import RAGChain, RAGResponse
from src.utils.persistence import load_vector_store, save_vector_store
from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents
from src.embeddings.huggingface import get_embedding_model
from config.settings import (
    DB_DIR,
    MIN_SIMILARITY_THRESHOLD,
    MAX_CONTEXT_DOCS,
    RETRIEVAL_K,
    EMBEDDING_MODEL_NAME  # noqa: F401 – kept for reference
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


class UploadResponse(BaseModel):
    """Response model for /upload endpoint."""
    filename: str
    chunks_added: int
    total_vectors: int
    message: str


class DocumentInfo(BaseModel):
    """Info about a single uploaded document."""
    name: str
    chunks: int


class DocumentsResponse(BaseModel):
    """Response model for /documents endpoint."""
    documents: List[DocumentInfo]
    total_documents: int
    total_vectors: int


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
vector_store: Optional[FAISSVectorStore] = None
embedder = None  # HuggingFaceEmbeddings instance (LangChain)
initialization_error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global rag_chain, vector_store, embedder, initialization_error

    try:
        print("Initializing RAG system...")

        # Load vector store
        print("Loading vector store...")
        vector_store = load_vector_store(DB_DIR, "rag_index")
        print(f"Vector store loaded successfully")

        # Initialize embedder (shared between retriever and upload handler)
        print("Initializing embedder...")
        embedder = get_embedding_model()
        print("Embedder initialized successfully")

        # Initialize retriever
        print("Initializing retriever...")
        retriever = SemanticRetriever(
            vector_store=vector_store,
            default_k=RETRIEVAL_K,
            default_min_score=MIN_SIMILARITY_THRESHOLD  # Filter out low-relevance chunks at retrieval time
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

        # Monkey-patch _check_confidence to always use the live threshold value.
        # This is needed because uvicorn --reload may not reload src/rag/chain.py
        # when only settings.py or app.py changes.
        def _patched_check_confidence(self_chain, retrieved_docs):
            if not retrieved_docs:
                return False
            max_score = max(doc.get('score', 0.0) for doc in retrieved_docs)
            threshold = MIN_SIMILARITY_THRESHOLD
            print(f"DEBUG confidence check: max_score={max_score:.4f}, threshold={threshold:.4f}, pass={max_score >= threshold}")
            return max_score >= threshold

        import types
        rag_chain._check_confidence = types.MethodType(_patched_check_confidence, rag_chain)
        print(f"RAG system initialized successfully! (threshold={MIN_SIMILARITY_THRESHOLD})")


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
                "Ensure phi3:mini model is available (ollama pull phi3:mini)"
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
            "/upload": "POST - Upload a PDF document",
            "/documents": "GET - List uploaded documents",
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


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document and add it to the vector store.

    The document will be:
    1. Loaded and text extracted page by page
    2. Split into overlapping chunks
    3. Embedded using the same model as the retriever
    4. Added to the FAISS vector store
    5. Persisted to disk so it survives restarts
    """
    global vector_store

    # Check system is ready
    if initialization_error:
        raise HTTPException(status_code=503, detail=f"RAG system not ready: {initialization_error}")
    if vector_store is None or embedder is None:
        raise HTTPException(status_code=503, detail="RAG system is still initializing")

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file to a temp location
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        print(f"Processing uploaded file: {file.filename} ({len(content)} bytes)")

        # Step 1: Load document (extract text page by page)
        doc_pages = load_document(tmp_path)
        if not doc_pages:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF. It may be scanned/image-only.")

        # Override source metadata to use the original filename (not the temp path)
        for page in doc_pages:
            page["metadata"]["source"] = file.filename

        print(f"Extracted {len(doc_pages)} pages from {file.filename}")

        # Step 2: Split into chunks
        chunks = split_documents(doc_pages)
        if not chunks:
            raise HTTPException(status_code=422, detail="No valid text chunks could be created from this PDF.")

        print(f"Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.embed_documents(texts)  # LangChain HuggingFaceEmbeddings API
        print(f"Generated {len(embeddings)} embeddings")

        # Step 4: Build metadata list for each chunk
        metadatas = [chunk.metadata for chunk in chunks]

        # Step 5: Add to vector store
        vector_store.add_embeddings(
            embeddings=embeddings,
            metadatas=metadatas,
            texts=texts
        )

        # Step 6: Persist updated vector store to disk
        save_vector_store(vector_store, DB_DIR, "rag_index", include_stats=False)
        print(f"Vector store saved. Total vectors: {vector_store.index.ntotal}")

        return UploadResponse(
            filename=file.filename,
            chunks_added=len(chunks),
            total_vectors=vector_store.index.ntotal,
            message=f"Successfully processed '{file.filename}' into {len(chunks)} chunks"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    """
    List all documents currently in the vector store.
    Returns unique filenames and their chunk counts.
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="RAG system is initializing")

    # Count chunks per source document
    doc_counts: Dict[str, int] = {}
    for vec_id, meta in vector_store.metadata_map.items():
        source = meta.get("source", "unknown")
        doc_counts[source] = doc_counts.get(source, 0) + 1

    documents = [
        DocumentInfo(name=name, chunks=count)
        for name, count in sorted(doc_counts.items())
    ]

    return DocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        total_vectors=vector_store.index.ntotal if vector_store.index else 0
    )


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Remove all chunks belonging to a document from the vector store.

    FAISS doesn't support true deletion, so we rebuild the index
    keeping only vectors whose source != filename.
    """
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=503, detail="RAG system is initializing")

    # Find all vector IDs that belong to this document
    ids_to_keep = {}
    ids_to_delete = []
    for vec_id, meta in vector_store.metadata_map.items():
        if meta.get("source") == filename:
            ids_to_delete.append(vec_id)
        else:
            ids_to_keep[vec_id] = meta

    if not ids_to_delete:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in the knowledge base"
        )

    chunks_removed = len(ids_to_delete)

    try:
        import faiss
        import numpy as np

        # Rebuild the FAISS index without the deleted document's vectors
        new_store = type(vector_store)(dimension=vector_store.dimension)
        new_store._create_index()  # Always initialize — even if result is an empty index

        if ids_to_keep:
            # Reconstruct vectors for kept IDs
            kept_ids_sorted = sorted(ids_to_keep.keys())
            vectors = np.array(
                [vector_store.index.reconstruct(int(i)) for i in kept_ids_sorted],
                dtype=np.float32
            )
            # Add them back (already normalized)
            new_store.index.add(vectors)
            # Restore metadata
            for new_idx, old_id in enumerate(kept_ids_sorted):
                new_store.metadata_map[new_idx] = ids_to_keep[old_id]
            new_store.id_counter = len(kept_ids_sorted)

        # Replace global vector store
        vector_store = new_store

        # CRITICAL: also update the retriever inside rag_chain — otherwise it still
        # searches the OLD in-memory index and returns results from deleted documents!
        if rag_chain is not None and rag_chain.retriever is not None:
            rag_chain.retriever.vector_store = new_store

        # Persist updated store
        save_vector_store(vector_store, DB_DIR, "rag_index", include_stats=False)

        return {
            "message": f"Deleted '{filename}' ({chunks_removed} chunks removed)",
            "chunks_removed": chunks_removed,
            "total_vectors": vector_store.index.ntotal if vector_store.index else 0
        }

    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


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
        # Always use the latest threshold from settings (in case it changed after startup)
        rag_chain.min_similarity = MIN_SIMILARITY_THRESHOLD
        # Execute query
        start_time = time.time()
        rag_response: RAGResponse = rag_chain.query(request.question)
        query_time = time.time() - start_time


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
        # Override has_answer: trust retrieval confidence over LLM's self-reported uncertainty.
        # Small models like tinyllama often say "I don't know" even with valid context.
        has_answer = (
            len(sources) > 0 and
            rag_response.confidence >= MIN_SIMILARITY_THRESHOLD
        )

        # If LLM generated a fallback phrase but we have good sources, build answer from sources
        FALLBACK_PHRASES = [
            "i don't have enough information",
            "i don't know",
            "cannot answer",
            "not enough information",
            "insufficient information",
        ]
        answer_text = rag_response.answer
        if has_answer and any(p in answer_text.lower() for p in FALLBACK_PHRASES):
            print(f"DEBUG: Replaced fallback answer with structured source text")
            top_sources = [s for s in sources[:3] if s.text]
            if top_sources:
                # Build a properly structured markdown response from source chunks
                filenames = list({s.metadata.get("source", "the document") for s in top_sources if s.metadata})
                src_label = filenames[0] if len(filenames) == 1 else "the uploaded documents"

                # Intro line
                lines = [f"Based on **{src_label}**, here is the relevant information:\n"]

                # Bullet points — one per source chunk, cleaned up
                for i, s in enumerate(top_sources, 1):
                    chunk = s.text.strip()
                    # Break chunk into sentences and list them cleanly
                    sentences = [sent.strip() for sent in chunk.replace("\n", " ").split(". ") if sent.strip()]
                    if sentences:
                        lines.append(f"- " + ". ".join(sentences[:4]) + ("." if not sentences[0].endswith(".") else ""))

                # Conclusion line
                lines.append(f"\n*The above information was retrieved directly from the document with a confidence score of {rag_response.confidence:.0%}.*")

                answer_text = "\n".join(lines)

        response = QueryResponse(
            answer=answer_text,
            sources=sources,
            confidence=rag_response.confidence,
            has_answer=has_answer,
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
