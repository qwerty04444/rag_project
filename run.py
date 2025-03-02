import os
from src.ingestion.text_processor import TextProcessor
from src.ingestion.image_processor import ImageProcessor
from src.ingestion.video_processor import VideoProcessor
from src.ingestion.web_scraper import WebScraper
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.storage import Storage
from src.retrieval.retriever import Retriever
from src.generation.llama_model import LlamaModel
from src.utils.logger import get_logger

# Initialize Logger
logger = get_logger(__name__)

def main():
    logger.info("Starting the RAG pipeline...")

    # Initialize Components
    text_processor = TextProcessor()
    image_processor = ImageProcessor()
    video_processor = VideoProcessor()
    web_scraper = WebScraper()
    embedding_generator = EmbeddingGenerator()
    storage = Storage()
    retriever = Retriever()
    llama_model = LlamaModel()

    # Example Workflow (Modify as needed)
    query = "Explain Quantum Computing."
    
    # Retrieval (Currently LLM-only since no files are added yet)
    context = retriever.retrieve(query)
    
    # Generate Response
    response = llama_model.generate(query, context)

    logger.info(f"Generated Response: {response}")

if __name__ == "__main__":
    main()
