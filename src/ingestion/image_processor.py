import os
import tempfile
from typing import List, Dict, Any, Optional
import moviepy.editor as mp
import speech_recognition as sr
from src.utils.logger import setup_logger
from src.utils.chunker import chunk_text
from src.config import TEMP_DIR

logger = setup_logger(__name__)

class VideoProcessor:
    """
    Processes video and audio files and extracts text through transcription.
    """
    
    def __init__(self):
        """Initialize the video processor."""
        self.recognizer = sr.Recognizer()
        logger.info("Video processor initialized")
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a video or audio file and extract transcribed text.
        
        Args:
            file_path: Path to the video/audio file
            
        Returns:
            List of document dictionaries with extracted text
        """
        logger.info(f"Processing video/audio file: {file_path}")
        
        video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        audio_formats = ['.mp3', '.wav', '.ogg', '.flac', '.aac']
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension in video_formats:
                return self._process_video(file_path)
            elif file_extension in audio_formats:
                return self._process_audio(file_path)
            else:
                logger.warning(f"Unsupported video/audio format: {file_extension}")
                return []
        except Exception as e:
            logger.error(f"Error processing video/audio file {file_path}: {str(e)}")
            return []
            
    def _process_video(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract audio from video and transcribe it."""
        logger.info(f"Extracting audio from video: {file_path}")
        
        # Create a unique temporary file path
        temp_audio_path = os.path.join(TEMP_DIR, f"{os.path.basename(file_path)}.wav")
        
        try:
            # Extract audio from video
            video = mp.VideoFileClip(file_path)
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Process the extracted audio
            result = self._process_audio(temp_audio_path)
            
            # Update metadata
            for doc in result:
                doc['metadata']['file_type'] = 'video'
                doc['metadata']['original_file'] = os.path.basename(file_path)
                doc['source'] = file_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting audio from video {file_path}: {str(e)}")
            return []
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
            
    def _process_audio(self, file_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio to text."""
        logger.info(f"Transcribing audio: {file_path}")
        
        try:
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                
                if not text.strip():
                    logger.warning(f"No text transcribed from audio: {file_path}")
                    return []
                
                chunks = chunk_text(text)
                return [{
                    'source': file_path,
                    'content': chunk,
                    'metadata': {
                        'file_type': 'audio',
                        'filename': os.path.basename(file_path),
                        'chunk_index': i
                    }
                } for i, chunk in enumerate(chunks)]
                
        except Exception as e:
            logger.error(f"Error transcribing audio {file_path}: {str(e)}")
            return []