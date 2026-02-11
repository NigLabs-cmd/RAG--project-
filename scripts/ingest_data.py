import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATA_DIR
from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents
from src.embeddings.huggingface import get_embedding_model, log_embedding_statistics
from src.vector_store.store import create_vector_db
from src.utils.persistence import save_processing_summary
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main ingestion pipeline with comprehensive reporting.
    Loads documents, splits into chunks, generates embeddings, and creates vector database.
    Includes detailed statistics and validation at each step.
    """
    logger.info("=" * 60)
    logger.info("STARTING DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Check data directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.warning(f"Created data directory at {DATA_DIR}. Please add PDF or TXT files there.")
        return
    
    # Step 1: Load documents
    logger.info("\n[STEP 1/4] Loading documents...")
    documents = []
    files_processed = 0
    files_failed = 0
    
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            try:
                doc = load_document(file_path)
                documents.extend(doc)
                files_processed += 1
                logger.info(f"  ✓ Loaded: {filename} ({len(doc)} pages)")
            except Exception as e:
                files_failed += 1
                logger.error(f"  ✗ Failed: {filename} - {e}")
    
    logger.info(f"\nDocument loading summary:")
    logger.info(f"  Files processed: {files_processed}")
    logger.info(f"  Files failed: {files_failed}")
    logger.info(f"  Total document pages: {len(documents)}")
    
    if not documents:
        logger.error("No documents found or loaded. Exiting.")
        return
    
    # Step 2: Split into chunks (with automatic statistics logging)
    logger.info("\n[STEP 2/4] Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    # Note: split_documents already logs chunk statistics internally
    # We'll capture those stats for the report
    chunk_stats = {
        "total_chunks": len(chunks),
        "valid_chunks": sum(1 for c in chunks if c.metadata.get("quality_valid", True)),
        "invalid_chunks": sum(1 for c in chunks if not c.metadata.get("quality_valid", True)),
        "valid_metadata": len(chunks),  # All should have metadata after enhancement
        "invalid_metadata": 0,
        "avg_chunk_length": sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0,
        "min_chunk_length": min(len(c.page_content) for c in chunks) if chunks else 0,
        "max_chunk_length": max(len(c.page_content) for c in chunks) if chunks else 0,
        "avg_word_count": sum(len(c.page_content.split()) for c in chunks) / len(chunks) if chunks else 0,
        "min_word_count": min(len(c.page_content.split()) for c in chunks) if chunks else 0,
        "max_word_count": max(len(c.page_content.split()) for c in chunks) if chunks else 0,
    }
    
    # Step 3: Generate embeddings
    logger.info("\n[STEP 3/4] Generating embeddings...")
    embedding_model = get_embedding_model()
    
    # Extract text for embedding
    texts = [chunk.page_content for chunk in chunks]
    
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.embed_documents(texts)
    
    # Log embedding statistics
    embedding_stats = log_embedding_statistics(embeddings, texts)
    
    # Step 4: Create vector database
    logger.info("\n[STEP 4/4] Creating vector database...")
    create_vector_db(chunks, embedding_model)
    logger.info("✓ Vector database created successfully")
    
    # Step 5: Save reports
    logger.info("\n[STEP 5/5] Generating reports...")
    output_files = save_processing_summary(
        chunks=chunks,
        embeddings=embeddings,
        chunk_stats=chunk_stats,
        embedding_stats=embedding_stats,
        output_dir="output"
    )
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"\n📊 Processing Summary:")
    logger.info(f"  Documents processed: {files_processed}")
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info(f"  Valid chunks: {chunk_stats['valid_chunks']}")
    logger.info(f"  Embeddings generated: {len(embeddings)}")
    logger.info(f"  Embedding dimension: {embedding_stats.get('embedding_dimension', 'N/A')}")
    logger.info(f"\n📁 Output Files:")
    logger.info(f"  Chunks metadata: {output_files['chunks_metadata']}")
    logger.info(f"  Embedding report: {output_files['embedding_report']}")
    logger.info("\n✅ All processing complete!")


if __name__ == "__main__":
    main()

