"""
End-to-End RAG Pipeline Demo

This script demonstrates the complete RAG workflow:
1. Load documents
2. Chunk text
3. Generate embeddings
4. Store in vector database
5. Perform semantic retrieval
6. Save and load persistence

Usage:
    python pipeline_demo.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embeddings.huggingface import get_embedding_model, EXPECTED_EMBEDDING_DIM
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval import SemanticRetriever, retrieve_context
from src.utils.persistence import (
    save_vector_store,
    load_vector_store,
    get_vector_store_info
)
from langchain_core.documents import Document
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Sample Documents
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "content": """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn from data without being explicitly programmed. It involves training 
        algorithms on datasets to recognize patterns and make predictions.
        
        There are three main types of machine learning:
        1. Supervised Learning - Uses labeled data to train models
        2. Unsupervised Learning - Finds patterns in unlabeled data
        3. Reinforcement Learning - Learns through trial and error
        
        Common applications include image recognition, natural language processing,
        recommendation systems, and autonomous vehicles.
        """,
        "metadata": {"source": "ml_guide.txt", "category": "ML", "topic": "fundamentals"}
    },
    {
        "content": """
        Python Programming for Data Science
        
        Python is the most popular programming language for data science due to its
        simplicity and powerful libraries. Key libraries include:
        
        - NumPy: Numerical computing with arrays
        - Pandas: Data manipulation and analysis
        - Matplotlib/Seaborn: Data visualization
        - Scikit-learn: Machine learning algorithms
        - TensorFlow/PyTorch: Deep learning frameworks
        
        Python's syntax is clean and readable, making it ideal for beginners while
        being powerful enough for advanced applications. The extensive ecosystem
        of packages makes it versatile for various data science tasks.
        """,
        "metadata": {"source": "python_ds.txt", "category": "Python", "topic": "data_science"}
    },
    {
        "content": """
        Introduction to Neural Networks
        
        Neural networks are computational models inspired by the human brain. They
        consist of layers of interconnected nodes (neurons) that process information.
        
        Architecture:
        - Input Layer: Receives raw data
        - Hidden Layers: Process and transform data
        - Output Layer: Produces final predictions
        
        Training Process:
        1. Forward propagation: Data flows through the network
        2. Loss calculation: Measure prediction error
        3. Backpropagation: Adjust weights to minimize error
        4. Iteration: Repeat until convergence
        
        Deep learning uses neural networks with many hidden layers to learn
        complex patterns in large datasets.
        """,
        "metadata": {"source": "neural_nets.txt", "category": "ML", "topic": "deep_learning"}
    },
    {
        "content": """
        Data Preprocessing Techniques
        
        Data preprocessing is crucial for building effective machine learning models.
        It involves cleaning and transforming raw data into a suitable format.
        
        Key Steps:
        1. Data Cleaning: Handle missing values, remove duplicates
        2. Feature Scaling: Normalize or standardize numerical features
        3. Encoding: Convert categorical variables to numerical
        4. Feature Engineering: Create new features from existing ones
        5. Data Splitting: Divide into training, validation, and test sets
        
        Proper preprocessing can significantly improve model performance and
        prevent issues like overfitting or biased predictions.
        """,
        "metadata": {"source": "preprocessing.txt", "category": "DataScience", "topic": "preprocessing"}
    },
    {
        "content": """
        Web Development with Python
        
        Python offers excellent frameworks for web development:
        
        Flask:
        - Lightweight and flexible
        - Great for small to medium applications
        - Minimal boilerplate code
        
        Django:
        - Full-featured framework
        - Includes ORM, admin panel, authentication
        - Follows MTV (Model-Template-View) pattern
        
        FastAPI:
        - Modern, fast framework
        - Built-in API documentation
        - Async support for high performance
        
        Python's web frameworks make it easy to build scalable web applications
        with clean, maintainable code.
        """,
        "metadata": {"source": "web_dev.txt", "category": "Web", "topic": "frameworks"}
    }
]


# ============================================================================
# Pipeline Functions
# ============================================================================

def create_sample_documents():
    """Create LangChain Document objects from sample data."""
    logger.info("="*60)
    logger.info("STEP 1: Creating Sample Documents")
    logger.info("="*60)
    
    documents = []
    for doc_data in SAMPLE_DOCUMENTS:
        doc = Document(
            page_content=doc_data["content"].strip(),
            metadata=doc_data["metadata"]
        )
        documents.append(doc)
    
    logger.info(f"✓ Created {len(documents)} documents")
    for i, doc in enumerate(documents):
        logger.info(f"  {i+1}. {doc.metadata['source']} ({doc.metadata['category']})")
    
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Chunking Documents")
    logger.info("="*60)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"✓ Created {len(chunks)} chunks")
    logger.info(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    return chunks


def generate_embeddings(chunks, embedding_model):
    """Generate embeddings for chunks."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Generating Embeddings")
    logger.info("="*60)
    
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)
    
    logger.info(f"✓ Generated {len(embeddings)} embeddings")
    logger.info(f"  Dimension: {len(embeddings[0])}")
    
    return embeddings, texts


def build_vector_store(embeddings, texts, chunks):
    """Build FAISS vector store."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Building Vector Store")
    logger.info("="*60)
    
    # Create vector store
    vector_store = FAISSVectorStore(dimension=EXPECTED_EMBEDDING_DIM)
    
    # Prepare metadata
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata.copy()
        metadata["chunk_id"] = i
        metadata["text_length"] = len(chunk.page_content)
        metadatas.append(metadata)
    
    # Add embeddings
    vector_store.add_embeddings(embeddings, metadatas, texts)
    
    stats = vector_store.get_statistics()
    logger.info(f"✓ Vector store built successfully")
    logger.info(f"  Total vectors: {stats['total_vectors']}")
    logger.info(f"  Dimension: {stats['dimension']}")
    
    return vector_store


def perform_retrieval(vector_store, queries):
    """Perform semantic retrieval on queries."""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Semantic Retrieval")
    logger.info("="*60)
    
    retriever = SemanticRetriever(vector_store, default_k=3)
    
    results_list = []
    for i, query in enumerate(queries):
        logger.info(f"\nQuery {i+1}: '{query}'")
        
        results = retriever.retrieve(query, k=3)
        results_list.append(results)
        
        logger.info(f"Retrieved {len(results)} results:")
        for j, result in enumerate(results):
            score = result['score']
            category = result['metadata'].get('category', 'N/A')
            text_preview = result['text'][:80] + "..." if len(result['text']) > 80 else result['text']
            logger.info(f"  {j+1}. [Score: {score:.4f}] [{category}] {text_preview}")
    
    return results_list


def save_and_load_demo(vector_store):
    """Demonstrate save/load functionality."""
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Persistence Demo")
    logger.info("="*60)
    
    save_dir = "data/vector_db"
    index_name = "demo_index"
    
    # Save
    logger.info("\nSaving vector store...")
    saved_files = save_vector_store(
        vector_store,
        save_dir,
        index_name,
        include_stats=True
    )
    
    logger.info(f"✓ Saved {len(saved_files)} files:")
    for key, path in saved_files.items():
        logger.info(f"  {key}: {path}")
    
    # Get info without loading
    logger.info("\nGetting vector store info...")
    info = get_vector_store_info(save_dir, index_name)
    logger.info(f"✓ Vector store info:")
    logger.info(f"  Dimension: {info['metadata']['dimension']}")
    logger.info(f"  Total vectors: {info['metadata']['total_vectors']}")
    logger.info(f"  Created: {info['metadata']['created_at']}")
    
    # Load
    logger.info("\nLoading vector store...")
    loaded_store = load_vector_store(save_dir, index_name, validate=True)
    
    logger.info(f"✓ Loaded successfully")
    logger.info(f"  Vectors: {loaded_store.index.ntotal}")
    
    return loaded_store


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run the complete RAG pipeline."""
    print("\n" + "="*60)
    print("RAG PIPELINE DEMO")
    print("="*60)
    print("\nThis demo shows the complete RAG workflow:")
    print("1. Document loading")
    print("2. Text chunking")
    print("3. Embedding generation")
    print("4. Vector store creation")
    print("5. Semantic retrieval")
    print("6. Persistence (save/load)")
    print("="*60)
    
    try:
        # Step 1: Create documents
        documents = create_sample_documents()
        
        # Step 2: Chunk documents
        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        
        # Step 3: Load embedding model and generate embeddings
        embedding_model = get_embedding_model()
        embeddings, texts = generate_embeddings(chunks, embedding_model)
        
        # Step 4: Build vector store
        vector_store = build_vector_store(embeddings, texts, chunks)
        
        # Step 5: Perform retrieval
        test_queries = [
            "What is machine learning?",
            "How to use Python for data analysis?",
            "Explain neural networks"
        ]
        results = perform_retrieval(vector_store, test_queries)
        
        # Step 6: Save and load demo
        loaded_store = save_and_load_demo(vector_store)
        
        # Verify loaded store works
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION: Testing Loaded Store")
        logger.info("="*60)
        
        test_query = "What are Python libraries for data science?"
        logger.info(f"\nTest query: '{test_query}'")
        
        context = retrieve_context(
            test_query,
            loaded_store,
            k=2,
            as_string=False
        )
        
        logger.info(f"✓ Retrieved {len(context)} documents from loaded store")
        for i, doc in enumerate(context):
            logger.info(f"  {i+1}. {doc.metadata.get('source', 'N/A')} (Score: {doc.metadata.get('retrieval_score', 0):.4f})")
        
        # Success summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print("\n✓ All steps completed successfully:")
        print(f"  ✓ Loaded {len(documents)} documents")
        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ Generated {len(embeddings)} embeddings")
        print(f"  ✓ Built vector store with {vector_store.index.ntotal} vectors")
        print(f"  ✓ Performed {len(test_queries)} semantic searches")
        print(f"  ✓ Saved and loaded vector store successfully")
        print("\nThe RAG system is ready to use!")
        print(f"Vector store saved at: data/vector_db/demo_index")
        
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
