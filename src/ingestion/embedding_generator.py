import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger
from src.config import EMBEDDING_MODEL

logger = setup_logger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text using a pre-trained model.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        logger.info(f"Initializing embedding generator with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def generate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            List of document dictionaries with added 'embedding' field
        """
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        if not documents:
            return []
        
        try:
            # Extract content from documents
            texts = [doc['content'] for doc in documents]
            
            # Generate embeddings in batches
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch_texts)
                embeddings.extend(batch_embeddings)
            
            # Add embeddings to documents
            for i, doc in enumerate(documents):
                doc['embedding'] = embeddings[i].tolist()
                
            return documents
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return documents