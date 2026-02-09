from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List, Dict

def split_documents(doc_list: List[Dict]):
    """
    Takes a list of dictionaries (from loader.py) and splits them into 
    LangChain Document chunks while preserving metadata.
    """
    # 1. Convert dictionaries to LangChain Document objects
    documents = [
        Document(page_content=d["content"], metadata=d["metadata"]) 
        for d in doc_list
    ]
    
    # 2. Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 3. Perform the split
    chunks = text_splitter.split_documents(documents)
    
    print(f"Processed {len(documents)} document pages.")
    print(f"Generated {len(chunks)} text chunks.")
    
    return chunks
