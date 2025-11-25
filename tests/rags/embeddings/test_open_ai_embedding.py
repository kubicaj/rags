from unittest.mock import MagicMock, patch

from rags.embeddings.open_ai_embedding import OpenAIEmbedding


# Test initialization uses correct api_key and embedding_model
def test_openai_embedding_init(monkeypatch):
    mock_client = MagicMock()
    with patch("rags.embeddings.open_ai_embedding.OpenAI", return_value=mock_client) as mock_openai:
        emb = OpenAIEmbedding(api_key="abc", embedding_model="model-x")
        mock_openai.assert_called_once_with(api_key="abc")
        assert emb.open_ai_client == mock_client
        assert emb.embedding_model == "model-x"

# Test embed returns embedding from OpenAI API
def test_openai_embedding_embed(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.embeddings.create.return_value = mock_response

    with patch("rags.embeddings.open_ai_embedding.OpenAI", return_value=mock_client):
        emb = OpenAIEmbedding(api_key="abc", embedding_model="model-x")
        result = emb.embed("hello")
        mock_client.embeddings.create.assert_called_once_with(input="hello", model="model-x")
        assert result == [0.1, 0.2, 0.3]

# Test embed handles empty string input
def test_openai_embedding_embed_empty(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[])]
    mock_client.embeddings.create.return_value = mock_response

    with patch("rags.embeddings.open_ai_embedding.OpenAI", return_value=mock_client):
        emb = OpenAIEmbedding(api_key="abc", embedding_model="model-x")
        result = emb.embed("")
        mock_client.embeddings.create.assert_called_once_with(input="", model="model-x")
        assert result == []

