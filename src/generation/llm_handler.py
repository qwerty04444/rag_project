from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LLMHandler:
    """
    Handler for managing multiple LLM models with fallback capabilities.
    """
    
    def __init__(self, primary_model, backup_model=None):
        """
        Initialize the LLM handler with primary and optional backup models.
        
        Args:
            primary_model: The main LLM model to use
            backup_model: Backup model to use if primary fails
        """
        self.primary_model = primary_model
        self.backup_model = backup_model
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the primary model, falling back to backup if needed.
        
        Args:
            prompt: The input prompt for generation
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If all models fail to generate a response
        """
        errors = []
        
        # Try primary model first
        try:
            logger.info("Generating response using primary model")
            return self.primary_model.generate(prompt, **kwargs)
        except Exception as e:
            error_msg = f"Primary model failed: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
        
        # Try backup model if primary fails and backup exists
        if self.backup_model is not None:
            try:
                logger.info("Falling back to backup model")
                return self.backup_model.generate(prompt, **kwargs)
            except Exception as e:
                error_msg = f"Backup model failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        # If we got here, all models failed
        raise Exception(f"All LLM models failed: {'; '.join(errors)}")