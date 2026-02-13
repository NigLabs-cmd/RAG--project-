"""
Semantic Retrieval System for RAG

This module provides high-level retrieval functionality:
- Query embedding generation
- Top-k semantic search
- Result formatting with metadata
- Configurable retrieval parameters
- Performance tracking
"""

import time
from typing import List, Dict, Optional, Any
import logging
from langchain_core.documents import Document

from src.embeddings.huggingface import get_embedding_model
from src.vector_store.faiss_store import FAISSVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    High-level semantic retrieval system.
    
    Combines embedding model and vector store for easy query processing.
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model=None,
        default_k: int = 5,
        default_min_score: Optional[float] = None
    ):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store: FAISS vector store instance
            embedding_model: Optional embedding model (will create if not provided)
            default_k: Default number of results to return
            default_min_score: Default minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or get_embedding_model()
        self.default_k = default_k
        self.default_min_score = default_min_score
        
        logger.info(f"Initialized SemanticRetriever (default_k={default_k})")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: Optional[float] = None,
        return_documents: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            k: Number of results to return (uses default if None)
            min_score: Minimum similarity score (uses default if None)
            return_documents: If True, return LangChain Document objects
            
        Returns:
            List of results with content, metadata, and scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Use defaults if not specified
        k = k or self.default_k
        min_score = min_score if min_score is not None else self.default_min_score
        
        # Track performance
        start_time = time.time()
        
        # Generate query embedding
        logger.info(f"Processing query: '{query[:100]}...'")
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            min_score=min_score
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"Retrieved {len(results)} results in {elapsed_time*1000:.2f}ms "
            f"(k={k}, min_score={min_score})"
        )
        
        # Format results
        if return_documents:
            return self._format_as_documents(results)
        else:
            return self._format_results(results, query, elapsed_time)
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        elapsed_time: float
    ) -> List[Dict[str, Any]]:
        """Format results with additional context."""
        formatted_results = []
        
        for result in results:
            formatted = {
                "id": result["id"],
                "score": result["score"],
                "rank": result["rank"],
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "query": query,
                "retrieval_time_ms": elapsed_time * 1000
            }
            formatted_results.append(formatted)
        
        return formatted_results
    
    def _format_as_documents(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Document]:
        """Format results as LangChain Document objects."""
        documents = []
        
        for result in results:
            # Extract text and metadata
            text = result.get("text", "")
            metadata = result.get("metadata", {}).copy()
            
            # Add retrieval metadata
            metadata["retrieval_score"] = result["score"]
            metadata["retrieval_rank"] = result["rank"]
            metadata["retrieval_id"] = result["id"]
            
            # Create Document
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: Optional[int] = None,
        min_score: Optional[float] = None,
        return_documents: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of results per query
            min_score: Minimum similarity score
            return_documents: If True, return LangChain Documents
            
        Returns:
            List of result lists (one per query)
        """
        logger.info(f"Processing {len(queries)} queries in batch")
        
        all_results = []
        for query in queries:
            results = self.retrieve(
                query=query,
                k=k,
                min_score=min_score,
                return_documents=return_documents
            )
            all_results.append(results)
        
        return all_results
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with retrieval system statistics
        """
        vector_stats = self.vector_store.get_statistics()
        
        stats = {
            "vector_store": vector_stats,
            "default_k": self.default_k,
            "default_min_score": self.default_min_score,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": vector_stats.get("dimension", 384)
        }
        
        return stats


def create_retriever(
    vector_store: FAISSVectorStore,
    k: int = 5,
    min_score: Optional[float] = None
) -> SemanticRetriever:
    """
    Convenience function to create a semantic retriever.
    
    Args:
        vector_store: FAISS vector store instance
        k: Default number of results
        min_score: Default minimum score threshold
        
    Returns:
        Configured SemanticRetriever instance
    """
    return SemanticRetriever(
        vector_store=vector_store,
        default_k=k,
        default_min_score=min_score
    )


def retrieve_context(
    query: str,
    vector_store: FAISSVectorStore,
    k: int = 5,
    min_score: Optional[float] = None,
    as_string: bool = False
) -> Any:
    """
    Quick retrieval function for getting context.
    
    Args:
        query: Query string
        vector_store: FAISS vector store
        k: Number of results
        min_score: Minimum score threshold
        as_string: If True, return concatenated text string
        
    Returns:
        Results list or concatenated string
    """
    retriever = create_retriever(vector_store, k, min_score)
    results = retriever.retrieve(query, return_documents=True)
    
    if as_string:
        # Concatenate all text content
        context_parts = [doc.page_content for doc in results]
        return "\n\n".join(context_parts)
    
    return results
