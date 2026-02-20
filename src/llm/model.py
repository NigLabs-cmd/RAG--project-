from langchain_ollama import ChatOllama
from config.settings import LLM_MODEL

def get_llm():
    """
    Initializes and returns the ChatOllama LLM instance.
    """
    print(f"Initializing Ollama model: {LLM_MODEL}...")
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,        # Deterministic output for factual Q&A
        num_predict=1024,     # Max output tokens — ensures complete answers
        num_ctx=4096,         # Context window size — enough for full document chunks
    )
    return llm
