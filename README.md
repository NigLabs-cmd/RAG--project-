# Local CPU-Only RAG System

A modular, Python-based Retrieval Augmented Generation (RAG) system running locally using Ollama and FAISS, complete with a FastAPI backend and a React (Vite) frontend.

## Features
- **Privacy-focused**: Runs entirely on your local machine.
- **Cost-effective**: Uses open-source models (Llama 3, Phi-3, TinyLlama).
- **Lightweight**: Optimized for 8GB RAM systems.
- **Grounded Answers**: Strict context-only answering with citations.
- **Smart Fallback**: Returns "I don't know" when confidence is low.
- **Modern UI**: Interactive React frontend with 3D animations and robust document management.

## What's New - Phase 5 ✨
- **Full-Stack Architecture**: Segmented into a FastAPI backend and a beautiful React frontend.
- **Interactive Landing Page**: Modern, animated 3D landing page built with React.
- **Document Upload API**: Upload and process PDFs directly from the web UI.
- **Streaming & Dynamic Context**: Real-time response generation (backend ready) and intelligent confidence scoring.
- **Improved Retrieval**: Threshold-based document filtering at retrieval time.

## Setup

### Prerequisites
1. **Python 3.10+** installed
2. **Node.js 18+** and npm installed
3. **Ollama** installed ([ollama.com](https://ollama.com))

### 1. Backend Setup
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama Setup
Start Ollama and pull your preferred model:
```bash
ollama serve
ollama pull tinyllama
```

## Running the Complete System

The system now runs as a client-server web application. Follow [HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for full details.

### 1. Start the Backend Server
Open a terminal and run:
```bash
cd backend
python app.py
```
- API Documentation available at: `http://localhost:8000/docs`
- Health Check available at: `http://localhost:8000/health`

### 2. Start the Frontend
Open a **new terminal** and run:
```bash
cd frontend
npm install
npm run dev
```
Navigate your browser to **http://localhost:5173** to see the interactive application!

## Legacy & Demo Usage

You can still use the CLI components if you prefer:
- **Interactive CLI**: `python main.py`
- **RAG Demo**: `python rag_demo.py` (Demonstrates semantic retrieval & confidence fallback)

## Project Structure
- `backend/`: FastAPI application (`app.py`), APIs for query and document upload.
- `frontend/`: React + Vite web application (interactive QA interface, landing page, animations).
- `src/`: Core pipeline logic (ingestion, embeddings, vector_store, retrieval, rag, llm).
- `config/`: Configuration settings and paths.
- `scripts/`: Standalone utilities and demo scripts.
- `data/`: Folder for manual document ingestion via CLI.
- `docs/`: Expanded documentation (see `HOW_TO_RUN.md`).

## RAG Pipeline Features

### Strict Prompt Engineering
- Answers ONLY from provided context, citing sources in a structured format (`[doc_XXX]`).
- Returns "I don't have enough information" when uncertain to prevent hallucinations.

### Citation & Confidence System
- Evaluates similarity score threshold (default: 0.5) to decide on answer quality.
- Automatic citation extraction validated against retrieved documents.

## Configuration

Edit `config/settings.py` to customize system parameters:
- `MIN_SIMILARITY_THRESHOLD`: Confidence threshold (e.g., 0.5)
- `MAX_CONTEXT_DOCS`: Max number of source chunks used (e.g., 3)
- `LLM_MODEL`: Ollama model name (default: `tinyllama` or `phi3:mini`)
- `CHUNK_SIZE`: Number of tokens per chunk (default: 500)

## Troubleshooting

**Ollama not found:**
- Make sure Ollama is running: `ollama serve`
- Pull the model: `ollama pull tinyllama`

**Backend Connection Issues:**
- Check that the FastAPI backend is running on `http://localhost:8000`
- See `docs/HOW_TO_RUN.md` for full troubleshooting steps.

**Low-quality answers:**
- Increase `MIN_SIMILARITY_THRESHOLD` for stricter filtering.
- Upload more relevant PDFs via the frontend UI.
- Try a more capable LLM model (e.g., `llama3` instead of `tinyllama`).
