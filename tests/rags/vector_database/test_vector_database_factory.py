from unittest.mock import patch

import pytest

from rags.vector_database.vector_database_factory import (
    S3VectorBucketConfig,
    VectorDatabaseFactory,
    create_default_s3_vector_bucket_config,
)


# Test create_default_s3_vector_bucket_config returns correct config
def test_create_default_s3_vector_bucket_config():
    config = create_default_s3_vector_bucket_config()
    assert isinstance(config, S3VectorBucketConfig)
    assert config.bucket_name == "kubica-vector-bucket"
    assert config.index_name == "kubica-vector-index"
    assert config.dataType == "float32"
    assert config.dimension == 3072
    assert config.distance_metric == "cosine"
    assert "content" in config.non_filterable_metadata_keys

# Test create_vector_database with all kwargs provided
def test_create_vector_database_with_kwargs():
    config = create_default_s3_vector_bucket_config()
    with patch("rags.vector_database.vector_database_factory.S3VectorBucketIndex") as mock_index:
        VectorDatabaseFactory.create_vector_database(
            database_type="s3_vector_bucket",
            s3_vector_db_config=config,
            aws_access_key_id="id",
            aws_secret_access_key="secret",
            region_name="region"
        )
        mock_index.assert_called_once()
        args, kwargs = mock_index.call_args
        assert kwargs["s3_vector_db_config"] == config
        assert kwargs["aws_access_key_id"] == "id"
        assert kwargs["aws_secret_access_key"] == "secret"
        assert kwargs["region_name"] == "region"

# Test create_vector_database uses environment variables if not provided
def test_create_vector_database_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env_secret")
    monkeypatch.setenv("AWS_REGION", "env_region")
    with patch("rags.vector_database.vector_database_factory.S3VectorBucketIndex") as mock_index:
        VectorDatabaseFactory.create_vector_database(database_type="s3_vector_bucket")
        mock_index.assert_called_once()
        args, kwargs = mock_index.call_args
        assert kwargs["aws_access_key_id"] == "env_id"
        assert kwargs["aws_secret_access_key"] == "env_secret"
        assert kwargs["region_name"] == "env_region"

# Test create_vector_database raises ValueError for unsupported type
def test_create_vector_database_unsupported_type():
    with pytest.raises(ValueError):
        VectorDatabaseFactory.create_vector_database(database_type="not_supported")

