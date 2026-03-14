# 🚀 How to Run the Complete RAG System

## Prerequisites

1. **Python 3.10+** installed
2. **Node.js 18+** and npm installed
3. **Ollama** installed and running
4. **Vector store** created (from Phase 3)

## Step-by-Step Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Vector Store Exists

If you haven't created the vector store yet:

```bash
python pipeline_demo.py
```

This will process documents and create the FAISS index.

### 3. Start Ollama (if not running)

```bash
ollama serve
```

In another terminal, ensure the model is available:

```bash
ollama pull tinyllama
```

### 4. Start the Backend Server

Open a terminal and run:

```bash
cd backend
python app.py
```

You should see:
```
Starting RAG Backend Server...
API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health
```

Keep this terminal open!

### 5. Start the Frontend

Open a **new terminal** and run:

```bash
cd frontend
npm run dev
```

You should see:
```
VITE v7.3.1  ready in XXX ms
➜  Local:   http://localhost:5173/
```

### 6. Open Your Browser

Navigate to: **http://localhost:5173**

You should see the RAG Question Answering System interface!

## Testing the System

### Quick Test

1. Type a question in the input field: "What is machine learning?"
2. Click "Ask"
3. You should see:
   - A response with confidence badge
   - Citations (if found)
   - Source documents with similarity scores
   - Query processing time

### Run Automated Tests

In a **new terminal**:

```bash
python test_phase5.py
```

This will run comprehensive tests including:
- Backend health checks
- Query validation
- Edge case handling
- Performance measurements

## Troubleshooting

### Backend won't start

**Error: "Failed to load vector store"**
- Solution: Run `python pipeline_demo.py` first

**Error: "Failed to initialize Ollama"**
- Solution: Start Ollama with `ollama serve`
- Ensure model exists: `ollama pull tinyllama`

### Frontend won't connect

**Error: "Cannot connect to backend"**
- Check backend is running on port 8000
- Visit http://localhost:8000/health to verify

**CORS errors**
- Backend already has CORS configured
- Try restarting both servers

### No answers returned

- Check if vector store has documents
- Verify Ollama is responding: `ollama list`
- Check backend logs for errors

## What to Expect

### Good Queries (Should Find Answers)
- "What is machine learning?"
- "How to use Python for data analysis?"
- "What is HTML used for?"

### Fallback Queries (Not in Dataset)
- "What is quantum computing?"
- "Explain blockchain technology"

These will return "I cannot answer..." with low confidence.

## Next Steps

1. **Add more documents** to `data/` folder
2. **Re-run pipeline**: `python pipeline_demo.py`
3. **Test new queries** related to your documents
4. **Customize the UI** in `frontend/src/`
5. **Improve prompts** in `src/rag/prompts.py`

## Architecture Overview

```
┌─────────────┐
│   Browser   │
│  (React UI) │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   FastAPI   │
│   Backend   │
└──────┬──────┘
       │
       ├─────────► Vector Store (FAISS)
       │
       └─────────► Ollama LLM
```

## Stopping the System

1. Press `Ctrl+C` in the frontend terminal
2. Press `Ctrl+C` in the backend terminal
3. Optionally stop Ollama: `Ctrl+C` in Ollama terminal

---

**Congratulations! Your RAG system is now fully operational! 🎉**
