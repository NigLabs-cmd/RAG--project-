from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME
from typing import List, Dict, Tuple
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected embedding dimension for all-MiniLM-L6-v2
EXPECTED_EMBEDDING_DIM = 384


def get_embedding_model():
    """
    Initializes and returns the Hugging Face embedding model.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Force CPU for compatibility
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info(f"✓ Embedding model loaded successfully")
    return embeddings


def validate_embeddings(embeddings: List[List[float]], texts: List[str]) -> Tuple[bool, str, Dict]:
    """
    Validate the quality and correctness of generated embeddings.
    
    Args:
        embeddings: List of embedding vectors
        texts: List of original text chunks (for validation)
        
    Returns:
        Tuple of (is_valid, reason, validation_details)
    """
    if not embeddings:
        return False, "No embeddings provided", {}
    
    if len(embeddings) != len(texts):
        return False, f"Embedding count ({len(embeddings)}) doesn't match text count ({len(texts)})", {}
    
    validation_details = {
        "total_embeddings": len(embeddings),
        "embedding_dimension": None,
        "all_same_dimension": True,
        "contains_nan": False,
        "contains_inf": False,
        "all_normalized": True,
        "dimension_mismatch": []
    }
    
    # Check each embedding
    for i, emb in enumerate(embeddings):
        # Check if embedding is valid
        if emb is None:
            return False, f"Embedding {i} is None", validation_details
        
        if not isinstance(emb, (list, np.ndarray)):
            return False, f"Embedding {i} is not a list or array", validation_details
        
        emb_array = np.array(emb)
        
        # Set expected dimension from first embedding
        if validation_details["embedding_dimension"] is None:
            validation_details["embedding_dimension"] = len(emb_array)
        
        # Check dimension consistency
        if len(emb_array) != validation_details["embedding_dimension"]:
            validation_details["all_same_dimension"] = False
            validation_details["dimension_mismatch"].append({
                "index": i,
                "expected": validation_details["embedding_dimension"],
                "actual": len(emb_array)
            })
        
        # Check for NaN values
        if np.any(np.isnan(emb_array)):
            validation_details["contains_nan"] = True
            return False, f"Embedding {i} contains NaN values", validation_details
        
        # Check for Inf values
        if np.any(np.isinf(emb_array)):
            validation_details["contains_inf"] = True
            return False, f"Embedding {i} contains Inf values", validation_details
        
        # Check if normalized (L2 norm should be ~1.0)
        norm = np.linalg.norm(emb_array)
        if not (0.99 <= norm <= 1.01):  # Allow small floating point errors
            validation_details["all_normalized"] = False
    
    # Check expected dimension for MiniLM-L6-v2
    if validation_details["embedding_dimension"] != EXPECTED_EMBEDDING_DIM:
        logger.warning(
            f"Embedding dimension {validation_details['embedding_dimension']} "
            f"doesn't match expected {EXPECTED_EMBEDDING_DIM} for {EMBEDDING_MODEL_NAME}"
        )
    
    # Check dimension consistency
    if not validation_details["all_same_dimension"]:
        return False, "Embeddings have inconsistent dimensions", validation_details
    
    return True, "Valid", validation_details


def log_embedding_statistics(embeddings: List[List[float]], texts: List[str]) -> Dict:
    """
    Calculate and log statistics about the generated embeddings.
    
    Args:
        embeddings: List of embedding vectors
        texts: List of original text chunks
        
    Returns:
        Dictionary containing embedding statistics
    """
    if not embeddings:
        logger.warning("No embeddings to analyze")
        return {}
    
    # Convert to numpy for easier analysis
    emb_array = np.array(embeddings)
    
    # Calculate statistics
    stats = {
        "total_embeddings": len(embeddings),
        "embedding_dimension": emb_array.shape[1] if len(emb_array.shape) > 1 else len(embeddings[0]),
        "avg_norm": float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
        "min_norm": float(np.min([np.linalg.norm(emb) for emb in embeddings])),
        "max_norm": float(np.max([np.linalg.norm(emb) for emb in embeddings])),
        "avg_value": float(np.mean(emb_array)),
        "std_value": float(np.std(emb_array)),
        "min_value": float(np.min(emb_array)),
        "max_value": float(np.max(emb_array)),
    }
    
    # Validate embeddings
    is_valid, reason, validation_details = validate_embeddings(embeddings, texts)
    stats["validation_passed"] = is_valid
    stats["validation_reason"] = reason
    stats.update(validation_details)
    
    # Log statistics
    logger.info("=" * 60)
    logger.info("EMBEDDING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total embeddings: {stats['total_embeddings']}")
    logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
    logger.info(f"Expected dimension: {EXPECTED_EMBEDDING_DIM}")
    
    if stats['embedding_dimension'] == EXPECTED_EMBEDDING_DIM:
        logger.info("✓ Dimension matches expected value")
    else:
        logger.warning(f"✗ Dimension mismatch! Expected {EXPECTED_EMBEDDING_DIM}, got {stats['embedding_dimension']}")
    
    logger.info(f"\nNorm statistics (should be ~1.0 for normalized embeddings):")
    logger.info(f"  Avg norm: {stats['avg_norm']:.4f}")
    logger.info(f"  Min norm: {stats['min_norm']:.4f}")
    logger.info(f"  Max norm: {stats['max_norm']:.4f}")
    
    if stats['all_normalized']:
        logger.info("  ✓ All embeddings are normalized")
    else:
        logger.warning("  ✗ Some embeddings are not normalized")
    
    logger.info(f"\nValue statistics:")
    logger.info(f"  Avg value: {stats['avg_value']:.4f}")
    logger.info(f"  Std value: {stats['std_value']:.4f}")
    logger.info(f"  Value range: [{stats['min_value']:.4f}, {stats['max_value']:.4f}]")
    
    logger.info(f"\nValidation:")
    if is_valid:
        logger.info(f"  ✓ {reason}")
    else:
        logger.error(f"  ✗ {reason}")
    
    logger.info("=" * 60)
    
    return stats

