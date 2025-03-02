import os
import pytest
import tempfile
from src.ingestion.text_processor import TextProcessor
from src.ingestion.image_processor import ImageProcessor
from src.ingestion.video_processor import VideoProcessor
from src.ingestion.binary_processor import BinaryProcessor
from src.ingestion.web_scraper import WebScraper
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.storage import MilvusStorage

@pytest.fixture
def temp_text_file():
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"This is a test document for testing the text processor.")
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_json_file():
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        f.write(b'{"test": "data", "items": [1, 2, 3]}')
    yield f.name
    os.unlink(f.name)

class TestTextProcessor:
    def test_process_txt(self, temp_text_file):
        processor = TextProcessor()
        results = processor.process(temp_text_file)
        assert len(results) > 0
        assert results[0]['content'] is not None
        assert results[0]['source'] == temp_text_file
        assert results[0]['metadata']['file_type'] == 'txt'
        
    def test_process_json(self, temp_json_file):
        processor = TextProcessor()
        results = processor.process(temp_json_file)
        assert len(results) > 0
        assert results[0]['content'] is not None
        assert results[0]['source'] == temp_json_file
        assert results[0]['metadata']['file_type'] == 'json'

class TestEmbeddingGenerator:
    def test_generate_embeddings(self):
        generator = EmbeddingGenerator()
        documents = [{'content': 'This is a test document.'}]
        results = generator.generate(documents)
        assert len(results) == 1
        assert 'embedding' in results[0]
        assert isinstance(results[0]['embedding'], list)
        assert len(results[0]['embedding']) > 0

class TestWebScraper:
    def test_process_url(self):
        scraper = WebScraper()
        results = scraper.process("https://example.com")
        assert len(results) > 0
        assert results[0]['content'] is not None
        assert results[0]['source'] == "https://example.com"
        assert results[0]['metadata']['file_type'] == 'web'