import speech_recognition as sr
from pydub import AudioSegment
import os

class VideoProcessor:
    """Converts audio/video speech to text using SpeechRecognition."""

    def __init__(self, audio_format="wav"):
        """Initialize audio format (default: wav)."""
        self.audio_format = audio_format
        self.recognizer = sr.Recognizer()

    def convert_audio(self, input_path, output_path):
        """Converts audio/video file to WAV format for processing."""
        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            return output_path
        except Exception as e:
            return f"Error converting file: {str(e)}"

    def extract_text(self, audio_path):
        """Extracts speech-to-text from an audio file."""
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)  # Using Google's STT API
                return text.strip()
        except Exception as e:
            return f"Error processing audio: {str(e)}"

# Test the module
if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Convert MP3 to WAV (if needed)
    wav_path = processor.convert_audio("sample_audio.mp3", "sample_audio.wav")
    
    # Extract text from WAV
    text = processor.extract_text(wav_path)
    
    print("Extracted Speech-to-Text:\n", text)
