import os
from typing import List, Dict, Any, Optional
from src.ingestion.text_processor import TextProcessor
from src.ingestion.image_processor import ImageProcessor
from src.ingestion.video_processor import VideoProcessor
from src.ingestion.binary_processor import BinaryProcessor
from src.ingestion.web_scraper import WebScraper
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.storage import MilvusStorage
from src.retrieval.retriever import Retriever
from src.generation.llm_handler import LLMHandler
from src.utils.logger import setup_logger
from src.utils.helper import get_file_extension, is_binary_file, get_supported_extensions
from src.config import MAX_DOCUMENTS_RETURNED

logger = setup_logger(__name__)

class RAGOrchestrator:
    """
    Orchestrates the entire RAG pipeline from ingestion to generation.
    """
    
    def __init__(self):
        """Initialize the RAG orchestrator."""
        # Initialize processors
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        self.binary_processor = BinaryProcessor()
        self.web_scraper = WebScraper()
        
        # Initialize embedding generator and storage
        self.embedding_generator = EmbeddingGenerator()
        self.storage = MilvusStorage()
        
        # Initialize retriever and generator
        self.retriever = Retriever()
        self.llm_handler = LLMHandler()
        
        # Get supported extensions
        self.supported_extensions = get_supported_extensions()
        
        logger.info("RAG Orchestrator initialized")
        
    def ingest(self, input_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Ingest documents from a directory or file.
        
        Args:
            input_path: Path to directory or file to ingest
            recursive: Whether to recursively ingest files in subdirectories
            
        Returns:
            Dict containing stats about ingestion process
        """
        logger.info(f"Ingesting from path: {input_path}")
        
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'processed_documents': 0,
            'by_type': {}
        }
        
        try:
            if os.path.isfile(input_path):
                # Process a single file
                self._process_file(input_path, stats)
            elif os.path.isdir(input_path):
                # Process a directory
                self._process_directory(input_path, recursive, stats)
            else:
                logger.error(f"Path not found: {input_path}")
                
            logger.info(f"Ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            return stats
    
    def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Ingest content from a URL.
        
        Args:
            url: URL to ingest
            
        Returns:
            Dict containing stats about ingestion process
        """
        logger.info(f"Ingesting from URL: {url}")
        
        stats = {
            'total_urls': 1,
            'processed_urls': 0,
            'failed_urls': 0,
            'processed_documents': 0
        }
        
        try:
            # Process the URL
            documents = self.web_scraper.process(url)
            
            if documents:
                # Generate embeddings
                documents_with_embeddings = self.embedding_generator.generate(documents)
                
                # Store in Milvus
                if self.storage.store(documents_with_embeddings):
                    stats['processed_urls'] += 1
                    stats['processed_documents'] += len(documents)
                else:
                    stats['failed_urls'] += 1
            else:
                stats['failed_urls'] += 1
                
            logger.info(f"URL ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during URL ingestion: {str(e)}")
            stats['failed_urls'] += 1
            return stats
    
    def process_query(self, query_text: str, max_docs: int = MAX_DOCUMENTS_RETURNED) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query_text: The query text
            max_docs: Maximum number of documents to retrieve
            
        Returns:
            Dict containing the response and retrieved documents
        """
        logger.info(f"Processing query: {query_text}")
        
        try:
            # Retrieve relevant documents
            documents = self.retriever.retrieve(query_text, max_docs)
            
            # Generate response using LLM
            response = self.llm_handler.generate(query_text, documents)
            
            return {
                'query': query_text,
                'response': response,
                'documents': documents
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': query_text,
                'response': "I encountered an error while processing your query.",
                'documents': []
            }
    
    def clear_data(self) -> bool:
        """
        Clear all ingested data.
        
        Returns:
            True if successful
        """
        logger.info("Clearing all data")
        
        try:
            return self.storage.clear()
        except Exception as e:
            logger.error(f"Error clearing data: {str(e)}")
            return False
    
    def _process_directory(self, directory_path: str, recursive: bool, stats: Dict[str, Any]):
        """Process all files in a directory."""
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path, stats)
                
            if not recursive:
                break
    
    def _process_file(self, file_path: str, stats: Dict[str, Any]):
        """Process a single file based on its type."""
        stats['total_files'] += 1
        
        try:
            # Determine the file type
            extension = get_file_extension(file_path)
            
            # Process the file based on its type
            documents = []
            
            if extension in self.supported_extensions['text']:
                # Process as text
                documents = self.text_processor.process(file_path)
                file_type = 'text'
            elif extension in self.supported_extensions['image']:
                # Process as image
                documents = self.image_processor.process(file_path)
                file_type = 'image'
            elif extension in self.supported_extensions['video'] or extension in self.supported_extensions['audio']:
                # Process as video/audio
                documents = self.video_processor.process(file_path)
                file_type = 'video/audio'
            elif is_binary_file(file_path):
                # Process as binary
                documents = self.binary_processor.process(file_path)
                file_type = 'binary'
            else:
                # Try processing as text by default
                documents = self.text_processor.process(file_path)
                file_type = 'unknown'
            
            # Update stats
            if file_type not in stats['by_type']:
                stats['by_type'][file_type] = {'processed': 0, 'failed': 0}
            
            if documents:
                # Generate embeddings
                documents_with_embeddings = self.embedding_generator.generate(documents)
                
                # Store in Milvus
                if self.storage.store(documents_with_embeddings):
                    stats['processed_files'] += 1
                    stats['processed_documents'] += len(documents)
                    stats['by_type'][file_type]['processed'] += 1
                else:
                    stats['failed_files'] += 1
                    stats['by_type'][file_type]['failed'] += 1
            else:
                stats['failed_files'] += 1
                stats['by_type'][file_type]['failed'] += 1
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            stats['failed_files'] += 1