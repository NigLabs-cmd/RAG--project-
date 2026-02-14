# Local CPU-Only RAG System

A modular, Python-based Retrieval Augmented Generation (RAG) system running locally using Ollama and FAISS.

## Features
- **Privacy-focused**: Runs entirely on your local machine.
- **Cost-effective**: Uses open-source models (Llama 3, Phi-3, TinyLlama).
- **Lightweight**: Optimized for 8GB RAM systems.
- **Grounded Answers**: Strict context-only answering with citations.
- **Smart Fallback**: Returns "I don't know" when confidence is low.

## What's New - Phase 4 ✨
- **RAG Pipeline**: End-to-end question answering with Ollama
- **Citation System**: Automatic source citation in `[doc_XXX]` format
- **Confidence-Based Fallback**: Refuses to answer with poor context
- **Edge Case Handling**: Graceful handling of errors and edge cases

## Setup

1. **Install Ollama**
   - Download from [ollama.com](https://ollama.com).
   - Pull a model: `ollama pull tinyllama`
   - Start Ollama: `ollama serve`

2. **Install Dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add Documents**
   - Place your PDF or TXT files in the `data/` folder.

## Usage

### Option 1: Interactive Interface
Run the main interface:
```bash
python main.py
```

- **Ingest Documents**: Select option 1 to process files and create the vector database.
- **Query**: Select option 2 to ask questions about your documents.

### Option 2: RAG Demo
Run the RAG pipeline demo:
```bash
python rag_demo.py
```

This demonstrates:
- Semantic document retrieval
- Context-only answering with citations
- Confidence-based fallback handling

### Option 3: Integration Tests
Run tests without Ollama:
```bash
python test_integration.py
```

## Project Structure
- `src/`: Core logic
  - `ingestion/`: Document loading and chunking
  - `embeddings/`: HuggingFace embedding models
  - `vector_store/`: FAISS vector database
  - `retrieval/`: Semantic search
  - `rag/`: RAG pipeline with Ollama (NEW)
  - `llm/`: Ollama LLM integration
- `config/`: Settings and paths
- `scripts/`: Standalone scripts
- `data/`: Your documents
- `vector_db/`: Persisted vector store

## RAG Pipeline Features

### Strict Prompt Engineering
- Answers ONLY from provided context
- Cites sources using `[doc_XXX]` format
- Returns "I don't have enough information" when uncertain

### Citation System
- Automatic citation extraction
- Validation against retrieved documents
- Coverage metrics

### Confidence-Based Fallback
- Similarity threshold: 0.5 (configurable)
- Refuses to answer with low-quality context
- Prevents hallucination

## Example Query

```
Query: What is machine learning?

Answer: Machine learning is a subset of artificial intelligence [doc_000].

Sources:
  - [doc_000] Machine learning is a subset of AI... (similarity: 0.85)

Confidence: 0.85
Citations: ['doc_000']
```

## Testing

Run the test suites:
```bash
# Core RAG tests
pytest test_rag.py -v

# Edge case tests
pytest test_rag_edge_cases.py -v

# Integration tests (no Ollama required)
python test_integration.py
```

## Configuration

Edit `config/settings.py` to customize:
- `MIN_SIMILARITY_THRESHOLD`: Confidence threshold (default: 0.5)
- `MAX_CONTEXT_DOCS`: Max documents in context (default: 3)
- `LLM_MODEL`: Ollama model name (default: tinyllama)
- `CHUNK_SIZE`: Text chunk size (default: 500)

## Troubleshooting

**Ollama not found:**
- Make sure Ollama is running: `ollama serve`
- Pull the model: `ollama pull tinyllama`

**Low-quality answers:**
- Increase `MIN_SIMILARITY_THRESHOLD` for stricter filtering
- Add more relevant documents to your database
- Try a larger LLM model

**Out of memory:**
- Use smaller model: `tinyllama` (1.1B params)
- Reduce `MAX_CONTEXT_DOCS`
- Reduce `CHUNK_SIZE`

