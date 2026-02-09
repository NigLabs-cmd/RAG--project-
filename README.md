# Local CPU-Only RAG System

A modular, Python-based Retrieval Augmented Generation (RAG) system running locally using Ollama and FAISS.

## Features
- **Privacy-focused**: Runs entirely on your local machine.
- **Cost-effective**: Uses open-source models (Llama 3, Phi-3).
- **Lightweight**: Optimized for 8GB RAM systems.

## Setup

1. **Install Ollama**
   - Download from [ollama.com](https://ollama.com).
   - Pull a model: `ollama pull llama3`

2. **Install Dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add Documents**
   - Place your PDF or TXT files in the `data/` folder.

## Usage

Run the main interface:
```bash
python main.py
```

- **Ingest Documents**: Select option 1 to process files and create the vector database.
- **Query**: Select option 2 to ask questions about your documents.

## Project Structure
- `src/`: Core logic (ingestion, embeddings, vector store, RAG).
- `config/`: Settings and paths.
- `scripts/`: Standalone scripts.
