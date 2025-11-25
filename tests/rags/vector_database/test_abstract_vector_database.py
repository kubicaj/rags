from unittest.mock import MagicMock, patch

from rags.vector_database.abstract_vector_database import VectorItem


# Test VectorItem initialization
def test_vector_item_init():
    # Test normal initialization
    key = "abc"
    embedding_vector = [0.1, 0.2, 0.3]
    metadata = {"foo": "bar"}
    item = VectorItem(key, embedding_vector, metadata)
    assert item.key == key
    assert item.embedding_vector == embedding_vector
    assert item.metadata == metadata

    # Test with empty metadata and vector
    item = VectorItem("k", [], {})
    assert item.key == "k"
    assert item.embedding_vector == []
    assert item.metadata == {}

# Test VectorItem.create_from_file_chunk
def test_create_from_file_chunk():
    # Mock FileChunk
    mock_chunk = MagicMock()
    mock_chunk.metadata = {"source": "file.txt"}
    mock_chunk.content = "chunk content"
    chunk_embedding = [1.0, 2.0, 3.0]

    # Patch uuid to return a fixed value
    with patch("rags.vector_database.abstract_vector_database.uuid") as mock_uuid:
        mock_uuid.uuid4.return_value.hex = "fixeduuid"
        item = VectorItem.create_from_file_chunk(mock_chunk, chunk_embedding)
        assert item.key == "fixeduuid"
        assert item.embedding_vector == chunk_embedding
        assert item.metadata["source"] == "file.txt"
        assert item.metadata["content"] == "chunk content"

# Test create_from_file_chunk with empty metadata
def test_create_from_file_chunk_empty_metadata():
    mock_chunk = MagicMock()
    mock_chunk.metadata = {}
    mock_chunk.content = ""
    chunk_embedding = []
    with patch("rags.vector_database.abstract_vector_database.uuid") as mock_uuid:
        mock_uuid.uuid4.return_value.hex = "uuid2"
        item = VectorItem.create_from_file_chunk(mock_chunk, chunk_embedding)
        assert item.key == "uuid2"
        assert item.embedding_vector == []
        assert item.metadata == {"content": ""}

