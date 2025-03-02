import os
import magic
import binascii
import struct
from typing import List, Dict, Any, Optional
from src.utils.logger import setup_logger
from src.utils.chunker import chunk_text

logger = setup_logger(__name__)

class BinaryProcessor:
    """
    Processes binary files and attempts to extract useful information.
    """
    
    def __init__(self):
        """Initialize the binary processor."""
        logger.info("Binary processor initialized")
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a binary file and extract metadata and any readable text.
        
        Args:
            file_path: Path to the binary file
            
        Returns:
            List of document dictionaries with extracted information
        """
        logger.info(f"Processing binary file: {file_path}")
        
        try:
            # Get file type using libmagic
            file_type = magic.from_file(file_path)
            mime_type = magic.from_file(file_path, mime=True)
            
            # Extract file size and metadata
            file_size = os.path.getsize(file_path)
            file_stats = os.stat(file_path)
            
            # Extract basic metadata info
            metadata = {
                'file_type': 'binary',
                'filename': os.path.basename(file_path),
                'detected_type': file_type,
                'mime_type': mime_type,
                'file_size': file_size,
                'created': file_stats.st_ctime,
                'modified': file_stats.st_mtime
            }
            
            # Extract readable strings if file is not too large
            readable_text = ""
            if file_size < 10 * 1024 * 1024:  # Less than 10MB
                readable_text = self._extract_readable_strings(file_path)
            
            # Build a textual representation
            content = f"Binary file: {os.path.basename(file_path)}\n"
            content += f"Type: {file_type}\n"
            content += f"MIME: {mime_type}\n"
            content += f"Size: {file_size} bytes\n\n"
            
            if readable_text:
                content += "Readable text found in the binary file:\n"
                content += readable_text
            
            chunks = chunk_text(content)
            return [{
                'source': file_path,
                'content': chunk,
                'metadata': {**metadata, 'chunk_index': i}
            } for i, chunk in enumerate(chunks)]
            
        except Exception as e:
            logger.error(f"Error processing binary file {file_path}: {str(e)}")
            return []
    
    def _extract_readable_strings(self, file_path: str, min_length: int = 4) -> str:
        """Extract readable ASCII strings from a binary file."""
        try:
            readable_strings = []
            current_string = ""
            
            with open(file_path, 'rb') as f:
                while True:
                    byte = f.read(1)
                    if not byte:
                        break
                        
                    # Check if byte is printable ASCII
                    if 32 <= ord(byte) <= 126:
                        current_string += byte.decode('ascii')
                    else:
                        if len(current_string) >= min_length:
                            readable_strings.append(current_string)
                        current_string = ""
            
            # Add the last string if it meets the minimum length
            if len(current_string) >= min_length:
                readable_strings.append(current_string)
                
            # Limit the number of strings to avoid overwhelming
            if len(readable_strings) > 100:
                return "\n".join(readable_strings[:100]) + "\n[more strings truncated]"
            else:
                return "\n".join(readable_strings)
                
        except Exception as e:
            logger.error(f"Error extracting strings from {file_path}: {str(e)}")
            return ""