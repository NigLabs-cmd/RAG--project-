import sys
import os

def check_imports():
    print("Checking imports...")
    try:
        import langchain
        import langchain_community
        import langchain_huggingface
        import langchain_ollama
        import faiss
        import sentence_transformers
        print("✅ All Python dependencies installed.")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    return True

def check_ollama():
    print("Checking Ollama connection...")
    # Simple check if port 11434 is open
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 11434))
    if result == 0:
        print("✅ Ollama is running.")
        sock.close()
        return True
    else:
        print("❌ Ollama is NOT running. Please start Ollama before proceeding.")
        return False

def main():
    print("--- Environment Verification ---")
    if check_imports() and check_ollama():
        print("\n✅ Environment is ready! run 'python main.py' to start.")
    else:
        print("\n❌ Environment setup incomplete. Please fix errors above.")

if __name__ == "__main__":
    main()
