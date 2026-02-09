from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME

def get_embedding_model():
    """
    Initializes and returns the Hugging Face embedding model.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Force CPU for compatibility
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings
