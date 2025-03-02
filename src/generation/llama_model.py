import json
import requests
from typing import List, Dict, Any, Optional
from src.utils.logger import setup_logger
from src.config import HUGGINGFACE_API_KEY, PRIMARY_LLM_MODEL

logger = setup_logger(__name__)

class LlamaModel:
    """
    Interface to the Llama-3.3-70B model via Hugging Face API.
    """
    
    def __init__(self, model_name: str = PRIMARY_LLM_MODEL):
        """
        Initialize the Llama model.
        
        Args:
            model_name: Name of the Llama model on Hugging Face
        """
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        logger.info(f"Llama model initialized with {model_name}")
        
    def generate(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate a response to the query based on retrieved documents.
        
        Args:
            query: The query text
            documents: List of retrieved document dictionaries
            
        Returns:
            Generated response text
        """
        try:
            # Extract content from documents
            context = "\n\n".join([
                f"Document {i+1} (Source: {doc.get('source', 'unknown')}): {doc.get('content', '')}"
                for i, doc in enumerate(documents)
            ])
            
            # Create prompt
            prompt = self._create_rag_prompt(query, context)
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 512,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "do_sample": True
                    }
                }
            )
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return ""
                
            # Parse response
            result = response.json()
            
            # Handle different response formats from Hugging Face
            if isinstance(result, list) and result:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                return str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error using Llama model: {str(e)}")
            return ""
            
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create a prompt for RAG using the retrieved context."""
        return f"""<|system|>
You are an intelligent AI assistant. Answer the user's question based on the provided context. If the context doesn't contain the relevant information, say that you don't know and avoid making up information.

Context:
{context}
</|system|>

<|user|>
{query}
</|user|>

<|assistant|>"""