import os
from typing import Literal

from rags.vector_database.abstract_vector_database import AbstractVectorDatabase
from rags.vector_database.s3_vector_bucket_index import S3VectorBucketIndex, S3VectorBucketConfig


class VectorDatabaseFactory:

    @staticmethod
    def create_vector_database(database_type: Literal["s3_vector_bucket"], **kwargs) -> AbstractVectorDatabase:
        """
        Factory method to get the appropriate vector database instance based on the configuration provided.

        Args:
            database_type (Literal["s3_vector_bucket"]): The type of vector database to create.
            kwargs : Additional keyword arguments for the vector database instance. Possible args for S3VectorBucketIndex:
                - s3_vector_db_config (S3VectorBucketConfig): Configuration for the S3 vector bucket index.
                Use default values if not provided.
                - aws_access_key_id (str): AWS access key ID. Use environment variables if key is not provided.
                - aws_secret_access_key (str): AWS secret access key. Use environment variables if key is not provided.
                - region_name (str): AWS region name. Use environment variables if region is not provided.

        Returns:
            AbstractVectorDatabase: An instance of the specified vector database type.
        """
        if database_type == "s3_vector_bucket":
            return S3VectorBucketIndex(
                s3_vector_db_config=kwargs.get('s3_vector_db_config', create_default_s3_vector_bucket_config()),
                aws_access_key_id=kwargs.get('aws_access_key_id', os.environ.get("AWS_ACCESS_KEY_ID")),
                aws_secret_access_key=kwargs.get('aws_secret_access_key', os.environ.get("AWS_SECRET_ACCESS_KEY")),
                region_name=kwargs.get('region_name', os.environ.get("AWS_REGION"))
            )
        else:
            raise ValueError(f"Unsupported vector database type: {database_type}")


# ----------------------------------------------------------------------------------------------------------------------
# Default S3 Vector Bucket Configuration
# ----------------------------------------------------------------------------------------------------------------------
def create_default_s3_vector_bucket_config() -> S3VectorBucketConfig:
    """
    Create a default S3VectorBucketConfig instance with predefined settings.
    Return:
        S3VectorBucketConfig: The default S3 vector bucket configuration.
    """
    return S3VectorBucketConfig(
        bucket_name="kubica-vector-bucket",
        index_name="kubica-vector-index",
        dataType="float32",
        dimension=3072,  # this should match the embedding dimension used
        distance_metric="cosine",
        non_filterable_metadata_keys=["content", "source"]
    )
