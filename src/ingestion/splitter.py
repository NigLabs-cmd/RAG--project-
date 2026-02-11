from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List, Dict, Tuple
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces/tabs
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_chunk_quality(chunk: Document) -> Tuple[bool, str]:
    """
    Validate the quality of a text chunk.
    
    Args:
        chunk: LangChain Document object
        
    Returns:
        Tuple of (is_valid, reason)
    """
    content = chunk.page_content
    
    # Check minimum length (avoid too-short chunks)
    if len(content) < 50:
        return False, f"Too short ({len(content)} chars)"
    
    # Check if chunk is mostly whitespace
    if len(content.strip()) < len(content) * 0.5:
        return False, "Mostly whitespace"
    
    # Check for reasonable word count
    word_count = len(content.split())
    if word_count < 10:
        return False, f"Too few words ({word_count})"
    
    # Check for excessive repetition (same word repeated many times)
    words = content.lower().split()
    if words:
        most_common_word = max(set(words), key=words.count)
        repetition_ratio = words.count(most_common_word) / len(words)
        if repetition_ratio > 0.5:
            return False, f"Excessive repetition ({repetition_ratio:.1%})"
    
    return True, "Valid"


def validate_metadata(metadata: Dict) -> Tuple[bool, str]:
    """
    Validate chunk metadata completeness and correctness.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Tuple of (is_valid, reason)
    """
    required_fields = ["source", "chunk_index", "char_start", "char_end", "processed_at"]
    
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in metadata]
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"
    
    # Validate chunk_index is non-negative
    if metadata.get("chunk_index", -1) < 0:
        return False, "Invalid chunk_index (must be >= 0)"
    
    # Validate character positions
    char_start = metadata.get("char_start", -1)
    char_end = metadata.get("char_end", -1)
    if char_start < 0 or char_end < 0:
        return False, "Invalid character positions (must be >= 0)"
    if char_end <= char_start:
        return False, "char_end must be greater than char_start"
    
    # Validate timestamp format
    try:
        datetime.fromisoformat(metadata.get("processed_at", ""))
    except (ValueError, TypeError):
        return False, "Invalid timestamp format"
    
    return True, "Valid"


def log_chunk_statistics(chunks: List[Document]) -> Dict:
    """
    Calculate and log statistics about the generated chunks.
    
    Args:
        chunks: List of LangChain Document objects
        
    Returns:
        Dictionary containing chunk statistics
    """
    if not chunks:
        logger.warning("No chunks to analyze")
        return {}
    
    # Calculate statistics
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    word_counts = [len(chunk.page_content.split()) for chunk in chunks]
    
    # Validate chunks
    valid_chunks = 0
    invalid_chunks = []
    for i, chunk in enumerate(chunks):
        is_valid, reason = validate_chunk_quality(chunk)
        if is_valid:
            valid_chunks += 1
        else:
            invalid_chunks.append((i, reason))
    
    # Validate metadata
    valid_metadata = 0
    invalid_metadata = []
    for i, chunk in enumerate(chunks):
        is_valid, reason = validate_metadata(chunk.metadata)
        if is_valid:
            valid_metadata += 1
        else:
            invalid_metadata.append((i, reason))
    
    stats = {
        "total_chunks": len(chunks),
        "valid_chunks": valid_chunks,
        "invalid_chunks": len(invalid_chunks),
        "valid_metadata": valid_metadata,
        "invalid_metadata": len(invalid_metadata),
        "avg_chunk_length": sum(chunk_lengths) / len(chunks),
        "min_chunk_length": min(chunk_lengths),
        "max_chunk_length": max(chunk_lengths),
        "avg_word_count": sum(word_counts) / len(word_counts),
        "min_word_count": min(word_counts),
        "max_word_count": max(word_counts),
    }
    
    # Log statistics
    logger.info("=" * 60)
    logger.info("CHUNK STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Valid chunks: {stats['valid_chunks']} ({stats['valid_chunks']/stats['total_chunks']*100:.1f}%)")
    logger.info(f"Invalid chunks: {stats['invalid_chunks']}")
    logger.info(f"Valid metadata: {stats['valid_metadata']} ({stats['valid_metadata']/stats['total_chunks']*100:.1f}%)")
    logger.info(f"Invalid metadata: {stats['invalid_metadata']}")
    logger.info(f"Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
    logger.info(f"Chunk length range: {stats['min_chunk_length']} - {stats['max_chunk_length']} chars")
    logger.info(f"Avg word count: {stats['avg_word_count']:.0f} words")
    logger.info(f"Word count range: {stats['min_word_count']} - {stats['max_word_count']} words")
    
    if invalid_chunks:
        logger.warning(f"Found {len(invalid_chunks)} invalid chunks:")
        for idx, reason in invalid_chunks[:5]:  # Show first 5
            logger.warning(f"  Chunk {idx}: {reason}")
        if len(invalid_chunks) > 5:
            logger.warning(f"  ... and {len(invalid_chunks) - 5} more")
    
    if invalid_metadata:
        logger.warning(f"Found {len(invalid_metadata)} chunks with invalid metadata:")
        for idx, reason in invalid_metadata[:5]:  # Show first 5
            logger.warning(f"  Chunk {idx}: {reason}")
        if len(invalid_metadata) > 5:
            logger.warning(f"  ... and {len(invalid_metadata) - 5} more")
    
    logger.info("=" * 60)
    
    return stats


def split_documents(doc_list: List[Dict]):
    """
    Takes a list of dictionaries (from loader.py) and splits them into 
    LangChain Document chunks while preserving metadata.
    Includes text cleaning, validation, enhanced metadata, and statistics logging.
    """
    # Get processing timestamp
    processing_timestamp = datetime.now().isoformat()
    
    # 1. Convert dictionaries to LangChain Document objects with cleaned text
    documents = [
        Document(
            page_content=clean_text(d["content"]), 
            metadata=d["metadata"]
        ) 
        for d in doc_list
    ]
    
    logger.info(f"Loaded {len(documents)} document pages for processing")
    
    # 2. Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 3. Perform the split
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Generated {len(chunks)} text chunks from {len(documents)} documents")
    
    # 4. Enhance metadata for each chunk
    char_position = 0
    for i, chunk in enumerate(chunks):
        # Add chunk index
        chunk.metadata["chunk_index"] = i
        
        # Add character positions
        chunk_length = len(chunk.page_content)
        chunk.metadata["char_start"] = char_position
        chunk.metadata["char_end"] = char_position + chunk_length
        char_position += chunk_length
        
        # Add processing timestamp
        chunk.metadata["processed_at"] = processing_timestamp
        
        # Add chunk quality info
        is_valid, reason = validate_chunk_quality(chunk)
        chunk.metadata["quality_valid"] = is_valid
        if not is_valid:
            chunk.metadata["quality_reason"] = reason
    
    logger.info(f"Enhanced metadata for all {len(chunks)} chunks")
    
    # 5. Log chunk statistics
    stats = log_chunk_statistics(chunks)
    
    return chunks

