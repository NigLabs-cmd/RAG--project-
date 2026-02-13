"""
FAISS Vector Store Implementation for RAG System

This module provides a complete FAISS-based vector store with:
- CPU-optimized indexing using cosine similarity
- Metadata mapping and management
- Persistence (save/load)
- Index statistics and validation
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search.
    
    Uses IndexFlatIP (Inner Product) for cosine similarity search.
    Embeddings are normalized before adding to ensure cosine similarity.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension (default 384 for MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = None
        self.metadata_map = {}  # Maps index ID to metadata
        self.id_counter = 0
        self.created_at = None
        self.version = "1.0"
        
        logger.info(f"Initializing FAISS vector store (dimension={dimension})")
    
    def _create_index(self):
        """Create a new FAISS index for cosine similarity."""
        # IndexFlatIP uses inner product, which equals cosine similarity for normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        self.created_at = datetime.now().isoformat()
        logger.info(f"✓ Created FAISS IndexFlatIP (dimension={self.dimension})")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length for cosine similarity.
        
        Args:
            embeddings: Array of embeddings (n_samples, dimension)
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add embeddings to the FAISS index with optional metadata.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts for each embedding
            texts: Optional list of text content for each embedding
            
        Returns:
            List of assigned IDs
        """
        if not embeddings:
            logger.warning("No embeddings provided to add")
            return []
        
        # Create index if it doesn't exist
        if self.index is None:
            self._create_index()
        
        # Convert to numpy array
        emb_array = np.array(embeddings, dtype=np.float32)
        
        # Validate dimension
        if emb_array.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {emb_array.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Normalize embeddings for cosine similarity
        emb_array = self._normalize_embeddings(emb_array)
        
        # Add to FAISS index
        start_id = self.id_counter
        self.index.add(emb_array)
        
        # Store metadata
        assigned_ids = []
        for i in range(len(embeddings)):
            current_id = start_id + i
            assigned_ids.append(current_id)
            
            # Build metadata entry
            metadata = {
                "id": current_id,
                "added_at": datetime.now().isoformat()
            }
            
            # Add user-provided metadata
            if metadatas and i < len(metadatas):
                metadata.update(metadatas[i])
            
            # Add text content if provided
            if texts and i < len(texts):
                metadata["text"] = texts[i]
            
            self.metadata_map[current_id] = metadata
        
        self.id_counter += len(embeddings)
        
        logger.info(f"✓ Added {len(embeddings)} embeddings to index (IDs: {start_id}-{self.id_counter-1})")
        
        return assigned_ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_score: Optional minimum similarity score (0-1)
            
        Returns:
            List of results with metadata and scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []
        
        # Convert to numpy and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        query_array = self._normalize_embeddings(query_array)
        
        # Limit k to index size
        k = min(k, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query_array, k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Skip if below minimum score
            if min_score is not None and score < min_score:
                continue
            
            # Get metadata
            metadata = self.metadata_map.get(int(idx), {})
            
            result = {
                "id": int(idx),
                "score": float(score),
                "rank": i + 1,
                "metadata": metadata
            }
            
            # Include text if available
            if "text" in metadata:
                result["text"] = metadata["text"]
            
            results.append(result)
        
        logger.info(f"✓ Found {len(results)} results for query (k={k})")
        
        return results
    
    def batch_search(
        self,
        query_embeddings: List[List[float]],
        k: int = 5,
        min_score: Optional[float] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: List of query embedding vectors
            k: Number of results per query
            min_score: Optional minimum similarity score
            
        Returns:
            List of result lists (one per query)
        """
        results = []
        for query_emb in query_embeddings:
            query_results = self.search(query_emb, k, min_score)
            results.append(query_results)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_metadata": len(self.metadata_map),
            "created_at": self.created_at,
            "version": self.version,
            "index_type": "IndexFlatIP (Cosine Similarity)",
            "is_trained": True,  # Flat index doesn't require training
        }
        
        return stats
    
    def save(self, directory: str, index_name: str = "faiss_index"):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            directory: Directory to save files
            index_name: Base name for index files
        """
        if self.index is None:
            raise ValueError("Cannot save: index is not initialized")
        
        # Create directory if needed
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, f"{index_name}.faiss")
        faiss.write_index(self.index, index_path)
        logger.info(f"✓ Saved FAISS index to: {index_path}")
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{index_name}_metadata.json")
        save_data = {
            "dimension": self.dimension,
            "id_counter": self.id_counter,
            "created_at": self.created_at,
            "version": self.version,
            "metadata_map": self.metadata_map,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved metadata to: {metadata_path}")
        logger.info(f"✓ Vector store saved successfully ({self.index.ntotal} vectors)")
    
    def load(self, directory: str, index_name: str = "faiss_index"):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            directory: Directory containing saved files
            index_name: Base name for index files
        """
        # Load FAISS index
        index_path = os.path.join(directory, f"{index_name}.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        logger.info(f"✓ Loaded FAISS index from: {index_path}")
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{index_name}_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        # Restore state
        self.dimension = save_data["dimension"]
        self.id_counter = save_data["id_counter"]
        self.created_at = save_data["created_at"]
        self.version = save_data.get("version", "1.0")
        
        # Convert string keys back to integers
        self.metadata_map = {int(k): v for k, v in save_data["metadata_map"].items()}
        
        logger.info(f"✓ Loaded metadata from: {metadata_path}")
        logger.info(f"✓ Vector store loaded successfully ({self.index.ntotal} vectors)")
        
        # Validate
        self._validate_loaded_index()
    
    def _validate_loaded_index(self):
        """Validate loaded index integrity."""
        if self.index.d != self.dimension:
            raise ValueError(
                f"Index dimension mismatch: expected {self.dimension}, got {self.index.d}"
            )
        
        if self.index.ntotal != len(self.metadata_map):
            logger.warning(
                f"Metadata count ({len(self.metadata_map)}) doesn't match "
                f"index size ({self.index.ntotal})"
            )
        
        logger.info("✓ Index validation passed")
    
    def clear(self):
        """Clear the index and metadata."""
        self.index = None
        self.metadata_map = {}
        self.id_counter = 0
        self.created_at = None
        logger.info("✓ Vector store cleared")
    
    def __repr__(self):
        """String representation of the vector store."""
        n_vectors = self.index.ntotal if self.index else 0
        return (
            f"FAISSVectorStore(dimension={self.dimension}, "
            f"vectors={n_vectors}, metadata={len(self.metadata_map)})"
        )
