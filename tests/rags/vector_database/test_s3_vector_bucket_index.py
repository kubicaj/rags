from unittest.mock import MagicMock, patch

import pytest

from rags.vector_database.s3_vector_bucket_index import (
    S3VectorBucketConfig,
    S3VectorBucketIndex,
    VectorItem,
)


# Helper config for tests
@pytest.fixture
def s3_config():
    return S3VectorBucketConfig(
        bucket_name="bucket",
        index_name="index",
        dataType="float32",
        dimension=3,
        distance_metric="cosine",
        non_filterable_metadata_keys=["content"]
    )

# Patch boto3 client for all tests
@pytest.fixture
def mock_s3_client():
    with patch("rags.vector_database.s3_vector_bucket_index.boto3.client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        # Add NotFoundException to exceptions
        mock_instance.exceptions.NotFoundException = Exception
        yield mock_instance

# Test __init__ sets up client and attributes
def test_init_sets_attributes(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(
        s3_vector_db_config=s3_config,
        aws_access_key_id="id",
        aws_secret_access_key="secret",
        region_name="region"
    )
    assert db.s3_vector_db_config == s3_config
    assert db.aws_access_key_id == "id"
    assert db.aws_secret_access_key == "secret"
    assert db.region_name == "region"
    assert db.s3_vector_client == mock_s3_client

# Test add_vectors calls put_vectors with correct arguments
def test_add_vectors_calls_put_vectors(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    vectors = [
        VectorItem("k1", [1.0, 2.0], {"foo": "bar"}),
        VectorItem("k2", [3.0, 4.0], {"baz": "qux"}),
    ]
    db.add_vectors(vectors)
    mock_s3_client.put_vectors.assert_called_once()
    args, kwargs = mock_s3_client.put_vectors.call_args
    assert kwargs["vectorBucketName"] == "bucket"
    assert kwargs["indexName"] == "index"
    assert kwargs["vectors"][0]["key"] == "k1"
    assert kwargs["vectors"][1]["key"] == "k2"
    assert kwargs["vectors"][0]["data"]["float32"] == [1.0, 2.0]

# Test query_vectors returns vectors from response
def test_query_vectors_returns_vectors(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    mock_s3_client.query_vectors.return_value = {"vectors": [{"key": "k"}]}
    result = db.query_vectors([1.0, 2.0], 5)
    assert result == [{"key": "k"}]
    mock_s3_client.query_vectors.assert_called_once()
    args, kwargs = mock_s3_client.query_vectors.call_args
    assert kwargs["queryVector"]["float32"] == [1.0, 2.0]
    assert kwargs["topK"] == 5

# Test query_vectors returns empty list if no vectors key
def test_query_vectors_returns_empty_if_no_vectors(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    mock_s3_client.query_vectors.return_value = {}
    result = db.query_vectors([1.0], 1)
    assert result == []

# Test create_index: bucket exists, index exists (skips creation)
def test_create_index_bucket_and_index_exist(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    # No exception means bucket and index exist
    with patch("rags.vector_database.s3_vector_bucket_index.logger") as mock_logger:
        db.create_index()
        mock_s3_client.get_vector_bucket.assert_called_once()
        mock_s3_client.get_index.assert_called_once()
        mock_logger.info.assert_any_call(f"Creating S3 Vector Bucket Index: {s3_config.index_name}")
        # Should not call create_vector_bucket or create_index
        mock_s3_client.create_vector_bucket.assert_not_called()
        mock_s3_client.create_index.assert_not_called()

# Test create_index: bucket does not exist, index does not exist
def test_create_index_bucket_and_index_not_exist(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    # Raise NotFoundException for bucket and index
    mock_s3_client.get_vector_bucket.side_effect = mock_s3_client.exceptions.NotFoundException
    mock_s3_client.get_index.side_effect = mock_s3_client.exceptions.NotFoundException
    with patch("rags.vector_database.s3_vector_bucket_index.logger"):
        db.create_index()
        mock_s3_client.create_vector_bucket.assert_called_once()
        mock_s3_client.create_index.assert_called_once()

# Test create_index: bucket exists, index does not exist
def test_create_index_bucket_exists_index_not_exist(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    mock_s3_client.get_index.side_effect = mock_s3_client.exceptions.NotFoundException
    with patch("rags.vector_database.s3_vector_bucket_index.logger"):
        db.create_index()
        mock_s3_client.create_vector_bucket.assert_not_called()
        mock_s3_client.create_index.assert_called_once()

# Test delete_index: bucket does not exist
def test_delete_index_bucket_not_exist(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    mock_s3_client.get_vector_bucket.side_effect = mock_s3_client.exceptions.NotFoundException
    with patch("rags.vector_database.s3_vector_bucket_index.logger") as mock_logger:
        db.delete_index()
        mock_logger.warning.assert_called()
        mock_s3_client.delete_index.assert_not_called()

# Test delete_index: index does not exist
def test_delete_index_index_not_exist(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    mock_s3_client.get_index.side_effect = mock_s3_client.exceptions.NotFoundException
    with patch("rags.vector_database.s3_vector_bucket_index.logger"):
        db.delete_index()
        mock_s3_client.delete_index.assert_not_called()

# Test delete_index: index exists
def test_delete_index_index_exists(s3_config, mock_s3_client):
    db = S3VectorBucketIndex(s3_config, "id", "secret", "region")
    with patch("rags.vector_database.s3_vector_bucket_index.logger"):
        db.delete_index()
        mock_s3_client.delete_index.assert_called_once()

