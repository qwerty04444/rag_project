import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face API
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_documents")

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM Models
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "meta-llama/Llama-3.3-70B")
BACKUP_LLM_MODEL = os.getenv("BACKUP_LLM_MODEL", "deepseek-ai/deepseek-coder-33b-instruct")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "rag_system.log")

# Processing Options
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_DOCUMENTS_RETURNED = int(os.getenv("MAX_DOCUMENTS_RETURNED", "5"))

# Temp Directory
TEMP_DIR = os.getenv("TEMP_DIR", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))