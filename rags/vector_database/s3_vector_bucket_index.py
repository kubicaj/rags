from loguru import logger

from dataclasses import dataclass
from typing import Literal

import boto3
from rags.vector_database.abstract_vector_database import AbstractVectorDatabase, VectorItem


# ----------------------------------------------------------------------------------------------------------------------
# S3 Vector Bucket Index Implementation
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class S3VectorBucketConfig:
    """
    Configuration for S3 Vector Bucket Index.
    """
    bucket_name: str
    index_name: str
    dataType: str
    dimension: int
    distance_metric: Literal['euclidean', 'cosine']
    non_filterable_metadata_keys: list[str]


# ----------------------------------------------------------------------------------------------------------------------
# S3 Vector Bucket Index Class
# ----------------------------------------------------------------------------------------------------------------------
class S3VectorBucketIndex(AbstractVectorDatabase):
    """
    S3VectorBucketIndex is a vector database implementation that uses an S3 bucket to store and manage vector data.
    """

    def __init__(self, s3_vector_db_config: S3VectorBucketConfig, aws_access_key_id: str, aws_secret_access_key: str,
                 region_name: str):
        """
        Initialize the S3VectorBucketIndex with the given S3 bucket details.

        Args:
            s3_vector_db_config (S3VectorBucketConfig): Configuration for the S3 vector bucket index.
            aws_access_key_id (str): AWS access key ID.
            aws_secret_access_key (str): AWS secret access key.
            region_name (str): AWS region name.
        """
        self.s3_vector_db_config = s3_vector_db_config
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name

        self.s3_vector_client = boto3.client(
            's3vectors',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )

    def add_vectors(self, vectors: list[VectorItem]):
        """
        Add vectors to the S3 vector bucket index.

        Args:
            vectors (list[VectorItem]): A list of VectorItem instances to add to the index.
        """
        self.s3_vector_client.put_vectors(
            vectorBucketName=self.s3_vector_db_config.bucket_name,
            indexName=self.s3_vector_db_config.index_name,
            vectors=[
                {
                    'key': vector.key,
                    'data': {
                        self.s3_vector_db_config.dataType: vector.embedding_vector
                    },
                    'metadata': vector.metadata
                } for vector in vectors
            ]
        )

    def query_vectors(self, query_vector: list[float], top_k: int) -> list[dict]:
        pass

    def create_index(self):
        """
        Create the vector index in the S3 vector bucket.

        First check if bucket exists. If not, create it with default encoding settings.
        Then create the vector index within the bucket.
        """
        logger.info(f"Creating S3 Vector Bucket Index: {self.s3_vector_db_config.index_name}")
        # Checking existence of bucket, create if not exists
        try:
            self.s3_vector_client.get_vector_bucket(
                vectorBucketName=self.s3_vector_db_config.bucket_name
            )
            logger.info(f"Vector bucket {self.s3_vector_db_config.bucket_name} already exists.")
        except self.s3_vector_client.exceptions.NotFoundException:
            logger.info(f"Vector bucket {self.s3_vector_db_config.bucket_name} does not exist. Creating new bucket.")
            self.s3_vector_client.create_vector_bucket(
                vectorBucketName=self.s3_vector_db_config.bucket_name,
                encryptionConfiguration={
                    'sseType': 'AES256'
                }
            )

        # check existence of index, create if not exists else skip creation
        try:
            self.s3_vector_client.get_index(
                vectorBucketName=self.s3_vector_db_config.bucket_name,
                indexName=self.s3_vector_db_config.index_name
            )
            logger.info(
                f"Vector index {self.s3_vector_db_config.index_name} already exists in bucket "
                f"{self.s3_vector_db_config.bucket_name}. Skipping index creation.")
            return
        except self.s3_vector_client.exceptions.NotFoundException:
            logger.info(
                f"Vector index {self.s3_vector_db_config.index_name} does not exist in bucket "
                f"{self.s3_vector_db_config.bucket_name}. Creating new index.")

        # Create vector index logic
        self.s3_vector_client.create_index(
            vectorBucketName=self.s3_vector_db_config.bucket_name,
            indexName=self.s3_vector_db_config.index_name,
            dataType=self.s3_vector_db_config.dataType,
            dimension=self.s3_vector_db_config.dimension,
            distanceMetric=self.s3_vector_db_config.distance_metric,
            metadataConfiguration={
                'nonFilterableMetadataKeys': self.s3_vector_db_config.non_filterable_metadata_keys
            }
        )

    def delete_index(self):
        """
        Delete the vector index from the S3 vector bucket. This method delete the index only. Bucket keep untouched.

        First check if bucket exists. If not then skip deletion and log warning.
        """
        logger.info(f"Deleting S3 Vector Bucket Index: {self.s3_vector_db_config.index_name}")
        # Checking existence of bucket
        try:
            self.s3_vector_client.get_vector_bucket(
                vectorBucketName=self.s3_vector_db_config.bucket_name
            )
            logger.info(f"Vector bucket {self.s3_vector_db_config.bucket_name} exists. Proceeding with index deletion.")
        except self.s3_vector_client.exceptions.NotFoundException:
            logger.warning(
                f"Vector bucket {self.s3_vector_db_config.bucket_name} does not exist. Skipping index deletion.")
            return

        # Delete vector index logic
        self.s3_vector_client.delete_index(
            vectorBucketName=self.s3_vector_db_config.bucket_name,
            indexName=self.s3_vector_db_config.index_name
        )
