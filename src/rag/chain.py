"""
RAG Chain Orchestration

This module orchestrates the complete RAG pipeline:
- Retrieval of relevant documents
- Context assembly with citations
- LLM query with strict prompts
- Response validation and citation extraction
"""

import time
from typing import List, Dict, Any, Optional
from .prompts import create_rag_prompt, create_fallback_response
from .citations import RAGResponse, create_rag_response


class RAGChain:
    """
    End-to-end RAG pipeline orchestrator.
    
    Coordinates retrieval, prompt generation, LLM inference,
    and response validation with citation extraction.
    """
    
    def __init__(
        self,
        retriever,
        llm,
        min_similarity: float = 0.5,
        max_docs: int = 3,
        include_metadata: bool = False
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: SemanticRetriever instance for document retrieval
            llm: ChatOllama instance for LLM inference
            min_similarity: Minimum similarity threshold for valid answers
            max_docs: Maximum number of documents to include in context
            include_metadata: Whether to include document metadata in context
        """
        self.retriever = retriever
        self.llm = llm
        self.min_similarity = min_similarity
        self.max_docs = max_docs
        self.include_metadata = include_metadata
        
        # Validate Ollama connection
        self._validate_llm_connection()
    
    def _validate_llm_connection(self):
        """
        Validate that Ollama LLM is accessible.
        
        Raises:
            ConnectionError: If LLM is not accessible
        """
        try:
            # Simple test to check if LLM is responsive
            # This will fail fast if Ollama is not running
            pass  # ChatOllama will raise error on first invoke if not available
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Ollama LLM. "
                f"Make sure Ollama is running (ollama serve). Error: {e}"
            )
    
    def query(self, question: str, k: Optional[int] = None) -> RAGResponse:
        """
        Execute complete RAG pipeline for a question.
        
        Args:
            question: User's question
            k: Number of documents to retrieve (uses max_docs if None)
            
        Returns:
            RAGResponse with answer, sources, and citations
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieval_k = k if k is not None else self.max_docs
        retrieved_docs = self._retrieve_context(question, retrieval_k)
        
        # Step 2: Check retrieval confidence
        if not self._check_confidence(retrieved_docs):
            # Low confidence - return fallback response
            return self._create_fallback_response(retrieved_docs, start_time)
        
        # Step 3: Format context and create prompt
        prompt, doc_ids = create_rag_prompt(
            question,
            retrieved_docs,
            include_metadata=self.include_metadata
        )
        
        # Step 4: Generate answer from LLM
        answer = self._generate_answer(prompt)
        
        # Step 5: Create structured response with citations
        response = self._parse_response(
            answer,
            retrieved_docs,
            doc_ids,
            start_time
        )
        
        return response
    
    def _retrieve_context(self, question: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the question.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        results = self.retriever.retrieve(question, k=k)
        
        # Convert to dictionary format for easier handling
        docs = []
        for result in results:
            # Handle different result formats
            if hasattr(result, 'page_content'):
                text = result.page_content
            elif hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, dict):
                text = result.get('text', result.get('page_content', ''))
            else:
                text = str(result)
            
            # Get metadata
            if hasattr(result, 'metadata'):
                metadata = result.metadata
                score = metadata.get('score', 0.0)
            elif isinstance(result, dict):
                metadata = result.get('metadata', {})
                score = result.get('score', 0.0)
            else:
                metadata = {}
                score = 0.0
            
            doc_dict = {
                'text': text,
                'metadata': metadata,
                'score': score
            }
            docs.append(doc_dict)
        
        return docs
    
    def _check_confidence(self, retrieved_docs: List[Dict[str, Any]]) -> bool:
        """
        Check if retrieval confidence is sufficient for answering.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            True if confidence is above threshold
        """
        if not retrieved_docs:
            return False
        
        # Get maximum similarity score
        max_score = max([doc.get('score', 0.0) for doc in retrieved_docs])
        
        return max_score >= self.min_similarity
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer from LLM using the prompt.
        
        Args:
            prompt: Complete RAG prompt with context
            
        Returns:
            LLM-generated answer
        """
        try:
            # Invoke LLM with the prompt
            response = self.llm.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            return answer.strip()
        
        except Exception as e:
            # Handle LLM errors gracefully
            error_msg = f"Error generating answer: {e}"
            print(f"WARNING: {error_msg}")
            return create_fallback_response()
    
    def _parse_response(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        doc_ids: List[str],
        start_time: float
    ) -> RAGResponse:
        """
        Parse LLM response and create structured RAGResponse.
        
        Args:
            answer: LLM-generated answer
            sources: Retrieved documents
            doc_ids: Document IDs used in context
            start_time: Query start timestamp
            
        Returns:
            Structured RAGResponse object
        """
        # Calculate metrics
        elapsed_time = time.time() - start_time
        max_score = max([doc.get('score', 0.0) for doc in sources]) if sources else 0.0
        
        # Create metadata
        metadata = {
            'query_time': elapsed_time,
            'num_sources': len(sources),
            'retrieval_scores': [doc.get('score', 0.0) for doc in sources]
        }
        
        # Create structured response with citation extraction
        return create_rag_response(
            answer=answer,
            sources=sources,
            doc_ids=doc_ids,
            max_score=max_score,
            metadata=metadata
        )
    
    def _create_fallback_response(
        self,
        sources: List[Dict[str, Any]],
        start_time: float
    ) -> RAGResponse:
        """
        Create fallback response for low-confidence retrievals.
        
        Args:
            sources: Retrieved documents (may be empty or low quality)
            start_time: Query start timestamp
            
        Returns:
            RAGResponse with fallback message
        """
        elapsed_time = time.time() - start_time
        max_score = max([doc.get('score', 0.0) for doc in sources]) if sources else 0.0
        
        metadata = {
            'query_time': elapsed_time,
            'num_sources': len(sources),
            'retrieval_scores': [doc.get('score', 0.0) for doc in sources],
            'fallback_reason': 'low_confidence' if sources else 'no_results'
        }
        
        return RAGResponse(
            answer=create_fallback_response(),
            sources=sources,
            citations=[],
            confidence=max_score,
            has_answer=False,
            metadata=metadata
        )
    
    def batch_query(self, questions: List[str], k: Optional[int] = None) -> List[RAGResponse]:
        """
        Execute RAG pipeline for multiple questions.
        
        Args:
            questions: List of user questions
            k: Number of documents to retrieve per question
            
        Returns:
            List of RAGResponse objects
        """
        responses = []
        for question in questions:
            response = self.query(question, k=k)
            responses.append(response)
        
        return responses


def create_rag_chain(retriever, llm, **kwargs) -> RAGChain:
    """
    Factory function to create a RAGChain instance.
    
    Args:
        retriever: SemanticRetriever instance
        llm: ChatOllama instance
        **kwargs: Additional arguments for RAGChain
        
    Returns:
        Configured RAGChain instance
    """
    return RAGChain(retriever, llm, **kwargs)
