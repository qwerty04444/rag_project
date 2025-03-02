from typing import List, Dict, Any, Optional
from src.ingestion.storage import MilvusStorage
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.utils.logger import setup_logger
from src.config import MAX_DOCUMENTS_RETURNED

logger = setup_logger(__name__)

class Retriever:
    """
    Retrieves relevant documents from the vector database based on a query.
    """
    
    def __init__(self):
        """Initialize the retriever."""
        self.embedding_generator = EmbeddingGenerator()
        self.storage = MilvusStorage()
        logger.info("Retriever initialized")
        
    def retrieve(self, query: str, top_k: int = MAX_DOCUMENTS_RETURNED) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: The query text
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of document dictionaries with content, source, metadata, and score
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        try:
            # Generate embedding for the query
            query_doc = [{'content': query}]
            query_with_embedding = self.embedding_generator.generate(query_doc)
            query_embedding = query_with_embedding[0]['embedding']
            
            # Search for relevant documents
            documents = self.storage.search(query_embedding, top_k)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []