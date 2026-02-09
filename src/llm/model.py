from langchain_ollama import ChatOllama
from config.settings import LLM_MODEL

def get_llm():
    """
    Initializes and returns the ChatOllama LLM instance.
    """
    print(f"Initializing Ollama model: {LLM_MODEL}...")
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,  # Deterministic output for factual Q&A
    )
    return llm
