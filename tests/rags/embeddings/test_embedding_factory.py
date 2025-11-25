from unittest.mock import MagicMock, patch

import pytest

from rags.embeddings.embedding_factory import EmbeddingFactory


# Test that factory creates OpenAIEmbedding with provided api_key
def test_create_openai_with_api_key(monkeypatch):
    mock_embedding = MagicMock()
    with patch("rags.embeddings.embedding_factory.OpenAIEmbedding", return_value=mock_embedding) as mock_cls:
        result = EmbeddingFactory.create("openai", open_ai_api_key="test-key")
        mock_cls.assert_called_once_with(open_ai_api_key="test-key")
        assert result == mock_embedding

# Test that factory creates OpenAIEmbedding with api_key from env if not provided
def test_create_openai_with_env(monkeypatch):
    mock_embedding = MagicMock()
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    with patch("rags.embeddings.embedding_factory.OpenAIEmbedding", return_value=mock_embedding) as mock_cls:
        result = EmbeddingFactory.create("openai")
        mock_cls.assert_called_once_with(open_ai_api_key="env-key")
        assert result == mock_embedding

# Test that factory raises ValueError for unsupported type
def test_create_unsupported_type():
    with pytest.raises(ValueError):
        EmbeddingFactory.create("unsupported")

