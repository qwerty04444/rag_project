import os
import json
from typing import List, Dict, Any, Optional
import PyPDF2
import docx
from src.utils.logger import setup_logger
from src.utils.chunker import chunk_text

logger = setup_logger(__name__)

class TextProcessor:
    """
    Processes text-based files (TXT, JSON, DOCX, PDF) and extracts text content.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        logger.info("Text processor initialized")
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a text-based file and extract content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document dictionaries with extracted text
        """
        logger.info(f"Processing text file: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt':
                return self._process_txt(file_path)
            elif file_extension == '.json':
                return self._process_json(file_path)
            elif file_extension == '.pdf':
                return self._process_pdf(file_path)
            elif file_extension == '.docx':
                return self._process_docx(file_path)
            else:
                logger.warning(f"Unsupported text file format: {file_extension}")
                return self._process_as_text(file_path)
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
            
    def _process_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a TXT file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        chunks = chunk_text(text)
        return [{
            'source': file_path,
            'content': chunk,
            'metadata': {
                'file_type': 'txt',
                'filename': os.path.basename(file_path),
                'chunk_index': i
            }
        } for i, chunk in enumerate(chunks)]
            
    def _process_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a JSON file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            text = json.dumps(data, indent=2)
        elif isinstance(data, list):
            text = '\n'.join(json.dumps(item, indent=2) for item in data)
        else:
            text = str(data)
        
        chunks = chunk_text(text)
        return [{
            'source': file_path,
            'content': chunk,
            'metadata': {
                'file_type': 'json',
                'filename': os.path.basename(file_path),
                'chunk_index': i
            }
        } for i, chunk in enumerate(chunks)]
            
    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a PDF file."""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"Page {page_num + 1}:\n{page_text}\n\n"
        
        chunks = chunk_text(text)
        return [{
            'source': file_path,
            'content': chunk,
            'metadata': {
                'file_type': 'pdf',
                'filename': os.path.basename(file_path),
                'chunk_index': i
            }
        } for i, chunk in enumerate(chunks)]
            
    def _process_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a DOCX file."""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        chunks = chunk_text(text)
        return [{
            'source': file_path,
            'content': chunk,
            'metadata': {
                'file_type': 'docx',
                'filename': os.path.basename(file_path),
                'chunk_index': i
            }
        } for i, chunk in enumerate(chunks)]
            
    def _process_as_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Process any file as plain text."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            chunks = chunk_text(text)
            return [{
                'source': file_path,
                'content': chunk,
                'metadata': {
                    'file_type': 'unknown_text',
                    'filename': os.path.basename(file_path),
                    'chunk_index': i
                }
            } for i, chunk in enumerate(chunks)]
        except Exception as e:
            logger.error(f"Failed to process {file_path} as text: {str(e)}")
            return []