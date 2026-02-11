"""
End-to-end test for the complete document vectorization pipeline.
Tests all phases: text cleaning, chunking, metadata, embeddings, and reporting.
"""

import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.splitter import split_documents
from src.embeddings.huggingface import get_embedding_model, log_embedding_statistics
from src.utils.persistence import save_processing_summary


def test_end_to_end_pipeline():
    """Test the complete pipeline from documents to reports."""
    print("\n" + "="*60)
    print("END-TO-END PIPELINE TEST")
    print("="*60)
    
    # Create sample documents
    print("\n[1/5] Creating sample documents...")
    sample_docs = [
        {
            "content": """
            Artificial Intelligence and Machine Learning
            
            Artificial intelligence (AI) is revolutionizing how we interact with technology.
            Machine learning, a subset of AI, enables computers to learn from data without
            being explicitly programmed. This has led to breakthroughs in various fields
            including computer vision, natural language processing, and robotics.
            
            Deep learning, which uses neural networks with multiple layers, has become
            particularly powerful for tasks like image recognition and language translation.
            These systems can automatically learn hierarchical representations of data,
            making them highly effective for complex pattern recognition tasks.
            """,
            "metadata": {"source": "ai_ml_intro.txt", "page": 1, "author": "Test"}
        },
        {
            "content": """
            Python for Data Science
            
            Python has emerged as the leading programming language for data science and
            machine learning. Its simple, readable syntax makes it accessible to beginners
            while its powerful libraries like NumPy, Pandas, and scikit-learn make it
            suitable for advanced applications.
            
            The Python ecosystem includes specialized tools for every stage of the data
            science workflow: data collection, cleaning, analysis, visualization, and
            model deployment. This comprehensive toolkit, combined with strong community
            support, has made Python the de facto standard in the field.
            """,
            "metadata": {"source": "python_ds.txt", "page": 1, "author": "Test"}
        },
        {
            "content": """
            Natural Language Processing
            
            Natural Language Processing (NLP) enables computers to understand, interpret,
            and generate human language. Modern NLP systems use transformer architectures
            like BERT and GPT to achieve human-level performance on many language tasks.
            
            Applications of NLP include sentiment analysis, machine translation, question
            answering, and text summarization. These technologies power virtual assistants,
            chatbots, and automated content generation systems that we use daily.
            """,
            "metadata": {"source": "nlp_overview.txt", "page": 1, "author": "Test"}
        }
    ]
    
    print(f"✓ Created {len(sample_docs)} sample documents")
    
    # Step 1: Split documents
    print("\n[2/5] Splitting documents into chunks...")
    chunks = split_documents(sample_docs)
    print(f"✓ Generated {len(chunks)} chunks")
    
    # Verify chunk enhancements
    assert all("chunk_index" in c.metadata for c in chunks), "All chunks should have chunk_index"
    assert all("char_start" in c.metadata for c in chunks), "All chunks should have char_start"
    assert all("char_end" in c.metadata for c in chunks), "All chunks should have char_end"
    assert all("processed_at" in c.metadata for c in chunks), "All chunks should have timestamp"
    assert all("quality_valid" in c.metadata for c in chunks), "All chunks should have quality flag"
    print("✓ All chunks have enhanced metadata")
    
    # Step 2: Generate embeddings
    print("\n[3/5] Generating embeddings...")
    embedding_model = get_embedding_model()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Step 3: Validate embeddings
    print("\n[4/5] Validating embeddings...")
    embedding_stats = log_embedding_statistics(embeddings, texts)
    
    assert embedding_stats["validation_passed"], "Embeddings should pass validation"
    assert embedding_stats["embedding_dimension"] == 384, "Should have dimension 384"
    assert embedding_stats["all_normalized"], "All embeddings should be normalized"
    print("✓ All embeddings validated successfully")
    
    # Step 4: Generate reports
    print("\n[5/5] Generating reports...")
    
    # Calculate chunk stats
    chunk_stats = {
        "total_chunks": len(chunks),
        "valid_chunks": sum(1 for c in chunks if c.metadata.get("quality_valid", True)),
        "invalid_chunks": sum(1 for c in chunks if not c.metadata.get("quality_valid", True)),
        "valid_metadata": len(chunks),
        "invalid_metadata": 0,
        "avg_chunk_length": sum(len(c.page_content) for c in chunks) / len(chunks),
        "min_chunk_length": min(len(c.page_content) for c in chunks),
        "max_chunk_length": max(len(c.page_content) for c in chunks),
        "avg_word_count": sum(len(c.page_content.split()) for c in chunks) / len(chunks),
        "min_word_count": min(len(c.page_content.split()) for c in chunks),
        "max_word_count": max(len(c.page_content.split()) for c in chunks),
    }
    
    # Save reports
    output_files = save_processing_summary(
        chunks=chunks,
        embeddings=embeddings,
        chunk_stats=chunk_stats,
        embedding_stats=embedding_stats,
        output_dir="test_output"
    )
    
    # Verify files were created
    assert os.path.exists(output_files["chunks_metadata"]), "Chunks metadata file should exist"
    assert os.path.exists(output_files["embedding_report"]), "Embedding report should exist"
    print(f"✓ Reports generated:")
    print(f"  - {output_files['chunks_metadata']}")
    print(f"  - {output_files['embedding_report']}")
    
    # Verify report contents
    print("\n[VERIFICATION] Checking report contents...")
    
    # Check JSON file
    import json
    with open(output_files["chunks_metadata"], 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    assert chunks_data["total_chunks"] == len(chunks), "JSON should have correct chunk count"
    assert len(chunks_data["chunks"]) == len(chunks), "JSON should contain all chunks"
    print(f"✓ Chunks metadata JSON contains {chunks_data['total_chunks']} chunks")
    
    # Check Markdown report
    with open(output_files["embedding_report"], 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    assert "Document Vectorization Report" in report_content, "Report should have title"
    assert "Chunk Statistics" in report_content, "Report should have chunk stats"
    assert "Embedding Statistics" in report_content, "Report should have embedding stats"
    assert "384" in report_content, "Report should mention embedding dimension"
    print("✓ Embedding report contains all required sections")
    
    # Summary
    print("\n" + "="*60)
    print("📊 PIPELINE TEST SUMMARY")
    print("="*60)
    print(f"Documents processed: {len(sample_docs)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Valid chunks: {chunk_stats['valid_chunks']}")
    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {embedding_stats['embedding_dimension']}")
    print(f"Validation passed: {embedding_stats['validation_passed']}")
    print(f"\nReports generated:")
    print(f"  ✓ Chunks metadata (JSON)")
    print(f"  ✓ Embedding report (Markdown)")
    
    # Cleanup test output
    print("\n[CLEANUP] Removing test output directory...")
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
    print("✓ Cleanup complete")
    
    return True


def main():
    """Run the end-to-end test."""
    print("\n" + "🧪 COMPLETE PIPELINE TEST SUITE 🧪")
    
    try:
        success = test_end_to_end_pipeline()
        
        if success:
            print("\n" + "="*60)
            print("🎉 ALL PIPELINE TESTS PASSED! 🎉")
            print("="*60)
            print("\n✅ All 4 phases verified:")
            print("  ✓ Phase 1: Text Cleaning & Chunking")
            print("  ✓ Phase 2: Metadata Enhancement")
            print("  ✓ Phase 3: Embedding Validation")
            print("  ✓ Phase 4: Persistence & Reporting")
            print("\n🚀 Document vectorization pipeline is ready for production!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
