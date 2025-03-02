# RAG System with LLaMA-3.3-70B

A comprehensive Retrieval-Augmented Generation (RAG) system that can process multiple file types (text, PDFs, images, videos, binaries) and use state-of-the-art language models to provide accurate responses.

## Features

- Multi-format document processing:
  - Text files (TXT, JSON, DOCX, PDF)
  - Images (with OCR capabilities)
  - Videos and audio (transcription)
  - Binary files
  - Web scraping
- Vector database storage using Milvus
- High-quality retrieval system
- Generation using LLaMA-3.3-70B with DeepSeek backup
- Modular pipeline design for easy customization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag_project.git
cd rag_project


rag_project/
│── src/
│   │── __init__.py
│   │── config.py                     # Configuration settings (API keys, paths, etc.)
│   │── main.py                       # Entry point for querying the RAG model
│   │── ingestion/
│   │   │── __init__.py
│   │   │── text_processor.py         # Handle TXT, JSON, DOCX, PDFs
│   │   │── image_processor.py        # Extract text from images (OCR)
│   │   │── video_processor.py        # Convert video/audio to text
│   │   │── binary_processor.py       # Handle binary files
│   │   │── web_scraper.py            # Scrape text from URLs
│   │   │── embedding_generator.py    # Convert extracted text to embeddings
│   │   │── storage.py                # Store embeddings in Milvus
│   │── retrieval/
│   │   │── __init__.py
│   │   │── retriever.py              # Fetch relevant documents from Milvus
│   │── generation/
│   │   │── __init__.py
│   │   │── llm_handler.py            # Handle LLM requests with fallback logic
│   │   │── llama_model.py            # Hugging Face API call to Llama-3.3-70B
│   │   │── deepseek_model.py         # Backup model implementation
│   │── utils/
│   │   │── __init__.py
│   │   │── logger.py                 # Logging utility
│   │   │── helper.py                 # Helper functions
│   │   │── chunker.py                # Text chunking strategies
│   │── pipeline/
│   │   │── __init__.py
│   │   │── orchestrator.py           # Orchestrate the entire RAG pipeline
│── tests/
│   │── test_ingestion.py
│   │── test_retrieval.py
│   │── test_generation.py
│   │── test_pipeline.py
│── requirements.txt                   # Dependencies
│── .env.example                       # Example environment variables
│── README.md                          # Project documentation
│── run.py                             # Run the full pipeline
│── setup.py                           # Package setup
│── notebooks/
    │── demo.ipynb                     # Demo notebook
    │── performance_tests.ipynb        # Performance testing