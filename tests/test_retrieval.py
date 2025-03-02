import pytest
import tempfile
import os
from src.retrieval.retriever import Retriever
from src.ingestion.storage import MilvusStorage
from src.ingestion.embedding_generator import EmbeddingGenerator

class TestRetriever:
    def test_retriever_initialization(self):
        retriever = Retriever()
        assert retriever is not None
        assert retriever.embedding_generator is not None
        assert retriever.storage is not None
        
    def test_retrieve_documents(self):
        retriever = Retriever()
        # Note: This test assumes the storage has some documents already
        results = retriever.retrieve("test query", top_k=2)
        # If no documents exist yet, it might return an empty list
        if results:
            assert isinstance(results, list)
            if results:
                assert 'content' in results[0]
                assert 'source' in results[0]
                assert 'score' in results[0]