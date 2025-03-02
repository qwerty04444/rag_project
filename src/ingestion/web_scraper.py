import os
import requests
import tempfile
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from src.utils.logger import setup_logger
from src.utils.chunker import chunk_text

logger = setup_logger(__name__)

class WebScraper:
    """
    Scrapes content from web pages.
    """
    
    def __init__(self):
        """Initialize the web scraper."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        logger.info("Web scraper initialized")
        
    def process(self, url: str) -> List[Dict[str, Any]]:
        """
        Process a URL and extract content.
        
        Args:
            url: URL to scrape
            
        Returns:
            List of document dictionaries with extracted content
        """
        logger.info(f"Scraping URL: {url}")
        
        try:
            # Send a GET request
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.text.strip() if soup.title else "No title"
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                tag.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                # Extract text and clean it
                text = main_content.get_text(separator='\n')
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                content = "\n".join(lines)
            else:
                content = soup.get_text(separator='\n')
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                content = "\n".join(lines)
            
            if not content.strip():
                logger.warning(f"No content extracted from URL: {url}")
                return []
            
            # Create a structured document
            document = f"Title: {title}\nURL: {url}\n\nContent:\n{content}"
            
            # Create domain-specific source identifier
            domain = urlparse(url).netloc
            
            chunks = chunk_text(document)
            return [{
                'source': url,
                'content': chunk,
                'metadata': {
                    'file_type': 'web',
                    'title': title,
                    'domain': domain,
                    'chunk_index': i
                }
            } for i, chunk in enumerate(chunks)]
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return []