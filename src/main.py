import os
import argparse
from typing import List, Dict, Any, Optional

from src.utils.logger import setup_logger
from src.pipeline.orchestrator import RAGOrchestrator
from src.config import MAX_DOCUMENTS_RETURNED

logger = setup_logger(__name__)

class RAGSystem:
    """
    Main class for the RAG system that provides an interface for users to interact with.
    """
    def __init__(self):
        """Initialize the RAG system."""
        self.orchestrator = RAGOrchestrator()
        logger.info("RAG System initialized")
        
    def ingest_documents(self, input_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Ingest documents from a directory or file.
        
        Args:
            input_path: Path to directory or file to ingest
            recursive: Whether to recursively ingest files in subdirectories
            
        Returns:
            Dict containing stats about ingestion process
        """
        logger.info(f"Ingesting documents from {input_path}")
        return self.orchestrator.ingest(input_path, recursive)
    
    def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Ingest content from a URL.
        
        Args:
            url: URL to ingest
            
        Returns:
            Dict containing stats about ingestion process
        """
        logger.info(f"Ingesting content from URL: {url}")
        return self.orchestrator.ingest_url(url)
    
    def query(self, query_text: str, max_docs: int = MAX_DOCUMENTS_RETURNED) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query_text: The query text
            max_docs: Maximum number of documents to retrieve
            
        Returns:
            Dict containing the response and retrieved documents
        """
        logger.info(f"Processing query: {query_text}")
        return self.orchestrator.process_query(query_text, max_docs)
    
    def clear_data(self) -> bool:
        """
        Clear all ingested data.
        
        Returns:
            True if successful
        """
        logger.info("Clearing all ingested data")
        return self.orchestrator.clear_data()


def main():
    """Command line interface for the RAG system."""
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument("--input", type=str, help="Input directory or file to ingest")
    parser.add_argument("--url", type=str, help="URL to ingest")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--clear", action="store_true", help="Clear all ingested data")
    
    args = parser.parse_args()
    
    rag = RAGSystem()
    
    if args.clear:
        rag.clear_data()
        
    if args.input:
        rag.ingest_documents(args.input)
        
    if args.url:
        rag.ingest_url(args.url)
        
    if args.query:
        result = rag.query(args.query)
        print("\nQuery:", args.query)
        print("\nResponse:", result["response"])
        print("\nRetrieved Documents:")
        for i, doc in enumerate(result["documents"]):
            print(f"{i+1}. {doc['source']} (Score: {doc['score']:.4f})")
            
    if not any([args.input, args.url, args.query, args.clear]):
        parser.print_help()


if __name__ == "__main__":
    main()