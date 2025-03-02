from typing import List
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better embedding and retrieval.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # If text is shorter than chunk_size, return as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get a chunk of size chunk_size or the remainder of the text
        end = min(start + chunk_size, len(text))
        
        # If we're not at the beginning of the text,
        # and not at the end of the text,
        # try to find a good breaking point
        if start > 0 and end < len(text):
            # Try to break at paragraph
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2  # Include the paragraph break
            else:
                # Try to break at line
                line_break = text.rfind('\n', start, end)
                if line_break != -1 and line_break > start + chunk_size // 2:
                    end = line_break + 1  # Include the line break
                else:
                    # Try to break at sentence
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2  # Include the period and space
                    else:
                        # Try to break at space
                        space_break = text.rfind(' ', start, end)
                        if space_break != -1 and space_break > start + chunk_size // 2:
                            end = space_break + 1  # Include the space
        
        # Add the chunk to our list
        chunks.append(text[start:end])
        
        # Move start position for next chunk, accounting for overlap
        start = end - chunk_overlap
        # Make sure we're making progress
        if start >= end:
            start = end
    
    return chunks