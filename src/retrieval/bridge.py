from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config.settings import RETRIEVAL_K

def get_rag_chain(llm, vector_db):
    """
    Creates a RetrievalQA chain using the LLM and Vector Store.
    """
    print("Building RAG chain...")
    
    # Custom prompt template for RAG
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Keep the answer concise.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain
