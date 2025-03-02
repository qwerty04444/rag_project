import os
import hashlib
from typing import List, Dict, Any, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_file_hash(file_path: str) -> str:
    """
    Calculate the hash of a file for deduplication and tracking.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hash string of the file
    """
    try:
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read and update hash in 64kb chunks for efficiency
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {str(e)}")
        return ""

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension in lowercase.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Lowercase file extension without the dot
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()[1:] if ext else ""

def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is binary rather than text.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is likely binary, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk  # A simple heuristic: if there's a null byte, it's likely binary
    except Exception as e:
        logger.error(f"Error checking if {file_path} is binary: {str(e)}")
        return False

def get_supported_extensions() -> Dict[str, List[str]]:
    """
    Get a dictionary of supported file extensions per processor.
    
    Returns:
        Dictionary mapping processor type to list of supported extensions
    """
    return {
        'text': ['txt', 'json', 'pdf', 'docx', 'md', 'html', 'csv', 'xml'],
        'image': ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'],
        'video': ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
        'audio': ['mp3', 'wav', 'ogg', 'flac', 'aac']
    }