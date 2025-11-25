from unittest.mock import MagicMock, patch

import pytest

from rags.rag_driver import RagDriver, RagQueryResult


# Test RagQueryResult __init__ and __repr__
def test_rag_query_result_init_and_repr():
    # Test initialization and string representation
    result = RagQueryResult("key1", 0.99, {"foo": "bar"})
    assert result.key == "key1"
    assert result.score == 0.99
    assert result.metadata == {"foo": "bar"}
    assert "RagQueryResult" in repr(result)
    assert "key1" in repr(result)
    assert "0.99" in repr(result)
    assert "'foo': 'bar'" in repr(result)

# Test RagDriver __init__ with mocked dependencies
@patch("rags.rag_driver.setup_logger")
@patch("rags.rag_driver.EmbeddingFactory")
@patch("rags.rag_driver.VectorDatabaseFactory")
def test_rag_driver_init(mock_vdb_factory, mock_embedding_factory, mock_setup_logger):
    # Test initialization with default embedding_type and vector_database_options
    mock_embedding = MagicMock()
    mock_embedding_factory.create.return_value = mock_embedding
    mock_vector_db = MagicMock()
    mock_vdb_factory.create_vector_database.return_value = mock_vector_db
    driver = RagDriver(embedding_type="openai", vector_database_options={})
    mock_setup_logger.assert_called_once()
    mock_embedding_factory.create.assert_called_with("openai")
    mock_vdb_factory.create_vector_database.assert_called_with(database_type="s3_vector_bucket")
    assert driver.embedding_instance == mock_embedding
    assert driver.vector_database == mock_vector_db

# Test _determine_file_type for various extensions
def test_determine_file_type():
    assert RagDriver._determine_file_type("foo.pdf") == "pdf"
    assert RagDriver._determine_file_type("bar.MD") == "md"
    assert RagDriver._determine_file_type("baz.txt ") == "txt"
    assert RagDriver._determine_file_type("noext") == "noext"

# Test _filter_files_by_type adds allowed types and warns on disallowed
@patch("rags.rag_driver.logger")
def test_filter_files_by_type_adds_and_warns(mock_logger):
    driver = RagDriver.__new__(RagDriver)
    allowed = ["pdf", "md"]
    files = []
    # Allowed type
    driver._filter_files_by_type("file1.pdf", allowed, files)
    assert "file1.pdf" in files
    # Disallowed type
    driver._filter_files_by_type("file2.txt", allowed, files)
    mock_logger.warning.assert_called_with("File type %s is not supported and will be skipped.", "txt")

# Test _create_chunks_from_file yields chunks from splitter
@patch("rags.rag_driver.logger")
@patch("rags.rag_driver.ChunkSplitterFactory")
def test_create_chunks_from_file_yields_chunks(mock_factory, mock_logger):
    # Mock splitter and its create_chunks
    mock_splitter = MagicMock()
    mock_splitter.create_chunks.return_value = ["chunk1", "chunk2"]
    mock_factory.create_based_on_file_type.return_value = mock_splitter
    files = ["file1.pdf", "file2.md"]
    gen = RagDriver._create_chunks_from_file(files)
    # Should yield a list for each file
    assert next(gen) == ["chunk1", "chunk2"]
    assert next(gen) == ["chunk1", "chunk2"]
    with pytest.raises(StopIteration):
        next(gen)

# Test find_in_rag returns list of RagQueryResult
@patch("rags.rag_driver.VectorDatabaseFactory")
@patch("rags.rag_driver.EmbeddingFactory")
def test_find_in_rag_returns_results(mock_embedding_factory, mock_vdb_factory):
    # Setup driver with mocks
    mock_embedding = MagicMock()
    mock_embedding.embed.return_value = [1.0, 2.0]
    mock_embedding_factory.create.return_value = mock_embedding
    mock_vector_db = MagicMock()
    mock_vector_db.query_vectors.return_value = [
        {"key": "k1", "distance": 0.1, "metadata": {"foo": "bar"}},
        {"key": "k2", "distance": 0.2, "metadata": {"baz": "qux"}},
    ]
    mock_vdb_factory.create_vector_database.return_value = mock_vector_db
    driver = RagDriver(embedding_type="openai", vector_database_options={})
    results = driver.find_in_rag("query", top_k=2)
    assert isinstance(results, list)
    assert all(isinstance(r, RagQueryResult) for r in results)
    assert results[0].key == "k1"
    assert results[1].score == 0.2

# Test fill_rag end-to-end logic with mocks
@patch("rags.rag_driver.logger")
@patch("rags.rag_driver.global_settings")
@patch("rags.rag_driver.os")
@patch("rags.rag_driver.VectorItem")
@patch("rags.rag_driver.ChunkSplitterFactory")
@patch("rags.rag_driver.EmbeddingFactory")
@patch("rags.rag_driver.VectorDatabaseFactory")
def test_fill_rag_logic(
    mock_vdb_factory, mock_embedding_factory, mock_chunk_factory, mock_vector_item,
    mock_os, mock_global_settings, mock_logger
):
    # Setup mocks
    mock_embedding = MagicMock()
    mock_embedding.embed.side_effect = lambda x: [1.0, 2.0]
    mock_embedding_factory.create.return_value = mock_embedding
    mock_vector_db = MagicMock()
    mock_vdb_factory.create_vector_database.return_value = mock_vector_db
    mock_vector_item.create_from_file_chunk.side_effect = lambda chunk, emb: f"vectoritem-{chunk.content}"
    # Setup os mocks
    mock_os.path.isfile.return_value = True
    mock_os.path.isdir.return_value = False
    # Setup chunk splitter
    mock_splitter = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "chunk-content"
    mock_splitter.create_chunks.return_value = [mock_chunk]
    mock_chunk_factory.create_based_on_file_type.return_value = mock_splitter
    # Setup global settings
    mock_global_settings.BATCH_VECTOR_UPSERT_SIZE = 2
    # Run fill_rag
    driver = RagDriver(embedding_type="openai", vector_database_options={})
    driver.fill_rag("somefile.pdf")
    # Should call delete_index and create_index
    mock_vector_db.delete_index.assert_called_once()
    mock_vector_db.create_index.assert_called_once()
    # Should calls add_vectors at least once
    assert mock_vector_db.add_vectors.called
    # Should log info about adding chunks
    assert mock_logger.info.call_count >= 1

# Test fill_rag raises ValueError for invalid source_path
@patch("rags.rag_driver.os")
def test_fill_rag_invalid_source_path(mock_os):
    mock_os.path.isfile.return_value = False
    mock_os.path.isdir.return_value = False
    driver = RagDriver.__new__(RagDriver)
    with pytest.raises(ValueError):
        driver.fill_rag("invalid_path")

