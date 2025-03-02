import pytest
import responses
from unittest.mock import patch, MagicMock
from src.generation.llama_model import LlamaModel
from src.generation.deepseek_model import DeepseekModel
from src.generation.llm_handler import LLMHandler

@pytest.fixture
def llama_model():
    with patch('src.generation.llama_model.HfApi') as mock_hf_api:
        model = LlamaModel(
            model_name="meta-llama/Llama-2-70b-chat-hf",
            api_key="test_key"
        )
        yield model

@pytest.fixture
def deepseek_model():
    with patch('src.generation.deepseek_model.pipeline') as mock_pipeline:
        model = DeepseekModel(
            model_name="deepseek-ai/deepseek-coder-33b-instruct"
        )
        yield model

@pytest.fixture
def llm_handler(llama_model, deepseek_model):
    handler = LLMHandler(
        primary_model=llama_model,
        backup_model=deepseek_model
    )
    return handler

@responses.activate
def test_llama_model_generation(llama_model):
    # Mock the Hugging Face API response
    responses.add(
        responses.POST,
        "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf",
        json={"generated_text": "This is a test response from Llama model"},
        status=200
    )
    
    response = llama_model.generate("Test prompt")
    assert response == "This is a test response from Llama model"

def test_deepseek_model_generation(deepseek_model):
    # Mock the pipeline response
    deepseek_model.pipeline.return_value = [{"generated_text": "This is a test response from Deepseek model"}]
    
    response = deepseek_model.generate("Test prompt")
    assert response == "This is a test response from Deepseek model"

def test_llm_handler_primary_success(llm_handler):
    with patch.object(llm_handler.primary_model, 'generate', return_value="Primary model response") as mock_primary:
        response = llm_handler.generate("Test prompt")
        mock_primary.assert_called_once()
        assert response == "Primary model response"
        
def test_llm_handler_fallback(llm_handler):
    with patch.object(llm_handler.primary_model, 'generate', side_effect=Exception("Primary model failed")) as mock_primary:
        with patch.object(llm_handler.backup_model, 'generate', return_value="Backup model response") as mock_backup:
            response = llm_handler.generate("Test prompt")
            mock_primary.assert_called_once()
            mock_backup.assert_called_once()
            assert response == "Backup model response"

def test_llm_handler_all_models_fail(llm_handler):
    with patch.object(llm_handler.primary_model, 'generate', side_effect=Exception("Primary model failed")):
        with patch.object(llm_handler.backup_model, 'generate', side_effect=Exception("Backup model failed")):
            with pytest.raises(Exception) as excinfo:
                llm_handler.generate("Test prompt")
            assert "All LLM models failed" in str(excinfo.value)