import time
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from src.utils.logger import setup_logger
from src.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION

logger = setup_logger(__name__)

class MilvusStorage:
    """
    Storage service using Milvus vector database.
    """
    
    def __init__(self, collection_name: str = MILVUS_COLLECTION):
        """
        Initialize Milvus storage.
        
        Args:
            collection_name: Name of the Milvus collection
        """
        self.collection_name = collection_name
        self.collection = None
        self._connect()
        self._init_collection()
        logger.info("Milvus storage initialized")
        
    def _connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default", 
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            logger.info(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
        
    def _init_collection(self):
        """Initialize Milvus collection."""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
                # Define collection schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # Dimension depends on model
                ]
                schema = CollectionSchema(fields)
                
                # Create collection
                self.collection = Collection(self.collection_name, schema)
                
                # Create index
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 8, "efConstruction": 64}
                }
                self.collection.create_index("embedding", index_params)
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Load collection
            self.collection.load()
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise
    
    def store(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Store documents in Milvus.
        
        Args:
            documents: List of document dictionaries with 'content', 'source', 'metadata', and 'embedding'
            
        Returns:
            True if successful
        """
        if not documents:
            return True
            
        try:
            # Prepare data for insertion
            sources = [doc.get('source', '') for doc in documents]
            contents = [doc.get('content', '') for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            embeddings = [doc.get('embedding', []) for doc in documents]
            
            # Ensure embedding dimensions are correct
            dim_size = len(embeddings[0])
            for i, emb in enumerate(embeddings):
                if len(emb) != dim_size:
                    logger.warning(f"Embedding dimension mismatch: {len(emb)} != {dim_size}")
                    embeddings[i] = [0.0] * dim_size
            
            # Insert data
            data = [sources, contents, metadatas, embeddings]
            self.collection.insert(data)
            self.collection.flush()
            logger.info(f"Successfully stored {len(documents)} documents in Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store documents in Milvus: {str(e)}")
            return False
    
    
       def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents in Milvus.
        
        Args:
            query_embedding: Embedding vector to search for
            top_k: Number of results to return
            
        Returns:
            List of document dictionaries with content, source, metadata, and score
        """
        try:
            # Ensure collection is loaded
            if not self.collection.is_loaded:
                self.collection.load()
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["source", "content", "metadata"]
            )
            
            # Format results
            documents = []
            for hits in results:
                for hit in hits:
                    documents.append({
                        'id': hit.id,
                        'content': hit.entity.get('content'),
                        'source': hit.entity.get('source'),
                        'metadata': hit.entity.get('metadata'),
                        'score': hit.score
                    })
            
            logger.info(f"Retrieved {len(documents)} documents from Milvus")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search documents in Milvus: {str(e)}")
            return []
    
    def clear(self) -> bool:
        """
        Clear all data in the collection.
        
        Returns:
            True if successful
        """
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped collection: {self.collection_name}")
                # Reinitialize collection
                self._init_collection()
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False