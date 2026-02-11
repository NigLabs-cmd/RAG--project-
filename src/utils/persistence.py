"""
Persistence utilities for saving chunks, metadata, and embedding reports.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_chunks_with_metadata(chunks: List[Document], output_dir: str = "output") -> str:
    """
    Save chunks with their metadata to a JSON file.
    
    Args:
        chunks: List of LangChain Document objects with metadata
        output_dir: Directory to save the output file
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chunks_metadata_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert chunks to serializable format
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunk_dict = {
            "chunk_id": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "content_length": len(chunk.page_content),
            "word_count": len(chunk.page_content.split())
        }
        chunks_data.append(chunk_dict)
    
    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "chunks": chunks_data
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(chunks)} chunks with metadata to: {filepath}")
    
    return filepath


def save_embedding_report(
    chunks: List[Document],
    embeddings: List[List[float]],
    chunk_stats: Dict,
    embedding_stats: Dict,
    output_dir: str = "output"
) -> str:
    """
    Generate and save a comprehensive embedding report.
    
    Args:
        chunks: List of LangChain Document objects
        embeddings: List of embedding vectors
        chunk_stats: Statistics from chunk processing
        embedding_stats: Statistics from embedding generation
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"embedding_report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Generate report content
    report_lines = []
    report_lines.append("# Document Vectorization Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n---\n")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append(f"- **Total Chunks:** {len(chunks)}")
    report_lines.append(f"- **Total Embeddings:** {len(embeddings)}")
    report_lines.append(f"- **Embedding Dimension:** {embedding_stats.get('embedding_dimension', 'N/A')}")
    report_lines.append(f"- **Processing Status:** {'✓ Success' if chunk_stats and embedding_stats else '✗ Failed'}")
    report_lines.append("")
    
    # Chunk Statistics
    if chunk_stats:
        report_lines.append("## Chunk Statistics")
        report_lines.append(f"- **Total Chunks:** {chunk_stats.get('total_chunks', 0)}")
        report_lines.append(f"- **Valid Chunks:** {chunk_stats.get('valid_chunks', 0)} "
                          f"({chunk_stats.get('valid_chunks', 0) / max(chunk_stats.get('total_chunks', 1), 1) * 100:.1f}%)")
        report_lines.append(f"- **Invalid Chunks:** {chunk_stats.get('invalid_chunks', 0)}")
        report_lines.append(f"- **Valid Metadata:** {chunk_stats.get('valid_metadata', 0)} "
                          f"({chunk_stats.get('valid_metadata', 0) / max(chunk_stats.get('total_chunks', 1), 1) * 100:.1f}%)")
        report_lines.append("")
        
        report_lines.append("### Chunk Length Statistics")
        report_lines.append(f"- **Average Length:** {chunk_stats.get('avg_chunk_length', 0):.0f} characters")
        report_lines.append(f"- **Min Length:** {chunk_stats.get('min_chunk_length', 0)} characters")
        report_lines.append(f"- **Max Length:** {chunk_stats.get('max_chunk_length', 0)} characters")
        report_lines.append("")
        
        report_lines.append("### Word Count Statistics")
        report_lines.append(f"- **Average Words:** {chunk_stats.get('avg_word_count', 0):.0f} words")
        report_lines.append(f"- **Min Words:** {chunk_stats.get('min_word_count', 0)} words")
        report_lines.append(f"- **Max Words:** {chunk_stats.get('max_word_count', 0)} words")
        report_lines.append("")
    
    # Embedding Statistics
    if embedding_stats:
        report_lines.append("## Embedding Statistics")
        report_lines.append(f"- **Total Embeddings:** {embedding_stats.get('total_embeddings', 0)}")
        report_lines.append(f"- **Embedding Dimension:** {embedding_stats.get('embedding_dimension', 0)}")
        report_lines.append(f"- **Expected Dimension:** 384 (MiniLM-L6-v2)")
        
        dim_match = embedding_stats.get('embedding_dimension', 0) == 384
        report_lines.append(f"- **Dimension Match:** {'✓ Yes' if dim_match else '✗ No'}")
        report_lines.append("")
        
        report_lines.append("### Normalization Statistics")
        report_lines.append(f"- **Average Norm:** {embedding_stats.get('avg_norm', 0):.4f}")
        report_lines.append(f"- **Min Norm:** {embedding_stats.get('min_norm', 0):.4f}")
        report_lines.append(f"- **Max Norm:** {embedding_stats.get('max_norm', 0):.4f}")
        report_lines.append(f"- **All Normalized:** {'✓ Yes' if embedding_stats.get('all_normalized', False) else '✗ No'}")
        report_lines.append("")
        
        report_lines.append("### Value Statistics")
        report_lines.append(f"- **Average Value:** {embedding_stats.get('avg_value', 0):.4f}")
        report_lines.append(f"- **Standard Deviation:** {embedding_stats.get('std_value', 0):.4f}")
        report_lines.append(f"- **Value Range:** [{embedding_stats.get('min_value', 0):.4f}, {embedding_stats.get('max_value', 0):.4f}]")
        report_lines.append("")
        
        report_lines.append("### Validation")
        validation_passed = embedding_stats.get('validation_passed', False)
        validation_reason = embedding_stats.get('validation_reason', 'Unknown')
        report_lines.append(f"- **Validation Status:** {'✓ Passed' if validation_passed else '✗ Failed'}")
        report_lines.append(f"- **Validation Reason:** {validation_reason}")
        report_lines.append("")
    
    # Sample Chunks
    report_lines.append("## Sample Chunks")
    report_lines.append("\nShowing first 3 chunks with metadata:\n")
    
    for i, chunk in enumerate(chunks[:3]):
        report_lines.append(f"### Chunk {i}")
        report_lines.append(f"**Content Preview:** {chunk.page_content[:200]}...")
        report_lines.append(f"\n**Metadata:**")
        for key, value in chunk.metadata.items():
            report_lines.append(f"- `{key}`: {value}")
        report_lines.append("")
    
    # Footer
    report_lines.append("---")
    report_lines.append("\n*Report generated by RAG Document Vectorization Pipeline*")
    
    # Write report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✓ Saved embedding report to: {filepath}")
    
    return filepath


def save_processing_summary(
    chunks: List[Document],
    embeddings: List[List[float]],
    chunk_stats: Dict,
    embedding_stats: Dict,
    output_dir: str = "output"
) -> Dict[str, str]:
    """
    Save both chunks metadata and embedding report.
    
    Args:
        chunks: List of LangChain Document objects
        embeddings: List of embedding vectors
        chunk_stats: Statistics from chunk processing
        embedding_stats: Statistics from embedding generation
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with paths to saved files
    """
    logger.info("=" * 60)
    logger.info("SAVING PROCESSING OUTPUTS")
    logger.info("=" * 60)
    
    # Save chunks metadata
    chunks_file = save_chunks_with_metadata(chunks, output_dir)
    
    # Save embedding report
    report_file = save_embedding_report(chunks, embeddings, chunk_stats, embedding_stats, output_dir)
    
    logger.info("=" * 60)
    logger.info("✓ All outputs saved successfully")
    logger.info("=" * 60)
    
    return {
        "chunks_metadata": chunks_file,
        "embedding_report": report_file
    }
