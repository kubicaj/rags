
import os
from typing import Generator, List, Literal

from loguru import logger

from rags import global_settings
from rags.chunks.abstract_splitter import AbstractFileSplitter, FileChunk
from rags.chunks.chunks_splitter_factory import ChunkSplitterFactory
from rags.common.setup_logger import setup_logger
from rags.embeddings.embedding_factory import EmbeddingFactory
from rags.vector_database.abstract_vector_database import VectorItem
from rags.vector_database.vector_database_factory import VectorDatabaseFactory


class RagQueryResult:
    """
    A class representing a single RAG query result.
    """

    def __init__(self, key: str, score: float, metadata: dict):
        """
        Initialize the RagQueryResult with the given key, score, and metadata.

        Args:
            key (str): The unique key for the vector item.
            score (float): The similarity score of the vector item.
            metadata (dict): The metadata associated with the vector item.
        """
        self.key = key
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        return (f"\nRagQueryResult("
                f"  key={self.key}, \n"
                f"  score={self.score}, \n"
                f"  metadata={self.metadata} \n"
                f")\n")


class RagDriver:
    """
    RAG (Retrieval-Augmented Generation) Driver class to manage the process of filling a RAG system

    Sample usage with default configuration:
        rag_driver = RagDriver(embedding_type="openai")
        rag_driver.fill_rag(source_path="/path/to/data/source")

    Sample usage with S3 vector bucket configuration:
        rag_driver = RagDriver(
            embedding_type="openai",
            vector_database_options={
                "s3_vector_db_config": S3VectorBucketConfig(
                    bucket_name="your-bucket-name",
                    index_name="your-index-name",
                    dataType="float32",
                    dimension=3072,  # this should match the embedding dimension used
                    distance_metric="cosine",
                    non_filterable_metadata_keys=["content", "source"]
                )

    Sample to query in the RAG system:
        result:RagQueryResult = RagDriver(vector_database_options={
        "s3_vector_db_config": S3VectorBucketConfig(
            bucket_name="kubica-vector-bucket",
            index_name="lake-formaation-knowladge",
            dataType="float32",
            dimension=3072,  # this should match the embedding dimension used
            distance_metric="cosine",
            non_filterable_metadata_keys=["content", "source"]
        )}).find_in_rag("What is Lake Formation?", top_k=5)
    """

    def __init__(self, embedding_type: Literal["openai"] = "openai", vector_database_options: dict = None):
        """
        Initialize the RAG driver with the specified embedding type.

        Args:
            embedding_type (Literal["openai"]): The embedding implementation to use. Default is "openai".
        """
        setup_logger()
        self.embedding_instance = EmbeddingFactory.create(embedding_type)
        self.vector_database = VectorDatabaseFactory.create_vector_database(
            database_type="s3_vector_bucket", **vector_database_options)

    @staticmethod
    def _determine_file_type(file_path: str) -> str:
        """
        Determine the file type based on the file extension.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The file type (extension) in lowercase.
        """
        return file_path.split('.')[-1].lower().rstrip()

    def _filter_files_by_type(self, file_path: str, allowed_file_types: list[str], file_to_process: list[str]):
        """
        Filter files by their types.

        Args:
            file_path (str): The path to the file.
            allowed_file_types (list[str]): List of allowed file types.
            file_to_process (list[str]): List to append the valid file paths.

        Returns:
            dict[str, list[str]]: A dictionary mapping file types to lists of file paths.
        """

        file_type = self._determine_file_type(file_path)
        if file_type in allowed_file_types:
            file_to_process.append(file_path)
        else:
            logger.warning("File type %s is not supported and will be skipped.", file_type)

    @staticmethod
    def _create_chunks_from_file(file_to_process: list[str]) -> Generator[List[FileChunk]]:
        """
        Create chunks from the list of files to process.
        Args:
            file_to_process (list[str]): List of file paths to process.
        Returns:
            Generator[List[FileChunk]: List of file chunks created from the files.
        """
        for file_path in file_to_process:
            logger.info(f"Processing file: {file_path}")
            splitter: AbstractFileSplitter = ChunkSplitterFactory.create_based_on_file_type(file_path)
            yield splitter.create_chunks()

    def find_in_rag(self, query: str, top_k: int) -> list[RagQueryResult]:
        """
        Find relevant information in the RAG system based on the query.

        Args:
            query (str): The input query string.
            top_k (int): The number of top relevant results to retrieve.

        Returns:
            list[dict]: A list of relevant results from the vector database.
        """
        query_embedding = self.embedding_instance.embed(query)
        results = self.vector_database.query_vectors(query_vector=query_embedding, top_k=top_k)
        return [
            RagQueryResult(
                key=result['key'],
                score=result['distance'],
                metadata=result['metadata']
            ) for result in results
        ]

    def fill_rag(self, source_path: str):
        """
        Fill the RAG system with data from the specified source path.

        Steps to do:
            1. look at the source path, determine file types, use appropriate chunk splitters to create chunks
            2. generate embeddings for the chunks using the configured embedding model
            3. store the chunks and their embeddings in the configured vector store

        Args:
            source_path (str): The path to the data source. It can be a file or directory. If a directory,
            all supported files within it will be processed.
        """

        # ------------------------------------------------
        # Step 1 - filter files and create chunks
        # ------------------------------------------------

        # filter files
        allowed_file_types = ['pdf', 'md']
        file_to_process: list[str] = []
        if os.path.isfile(source_path):
            self._filter_files_by_type(source_path, allowed_file_types, file_to_process)
        elif os.path.isdir(source_path):
            for root, _, files in os.walk(source_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self._filter_files_by_type(file_path, allowed_file_types, file_to_process)
        else:
            raise ValueError(f"Source path {source_path} is neither a file nor a directory.")

        # create chunks
        chunks_to_process: Generator[List[FileChunk]] = self._create_chunks_from_file(file_to_process)

        # ------------------------------------------------
        # Step 2 and 3 - generate embeddings and store in vector database
        # ------------------------------------------------

        # recreate the vector database index
        self.vector_database.delete_index()
        self.vector_database.create_index()

        embedding_batch: list[VectorItem] = []
        total_added_chunks = 0
        total_looped_chunks = 0
        logger.info("Starting to add chunks to the vector database...")
        there_is_next_file_to_process = True
        while there_is_next_file_to_process:
            # default chunks is empty list
            chunk_list = next(chunks_to_process, [])
            # set flag to false if no more files to process to end the while loop
            if not chunk_list:
                there_is_next_file_to_process = False
            # one chunk list corresponds to one file
            for chunk in chunk_list:
                embedding = self.embedding_instance.embed(chunk.content)
                # add to batch
                embedding_batch.append(VectorItem.create_from_file_chunk(chunk, embedding))
                # upsert batch if size reached or last chunk
                if len(embedding_batch) >= global_settings.BATCH_VECTOR_UPSERT_SIZE:
                    self.vector_database.add_vectors(vectors=embedding_batch)
                    total_added_chunks += len(embedding_batch)
                    logger.info(f"Added {total_added_chunks} chunks to the vector database.")
                    embedding_batch = []
                total_looped_chunks += 1
        # upsert any remaining embeddings in the batch
        if embedding_batch:
            self.vector_database.add_vectors(vectors=embedding_batch)
            total_added_chunks += len(embedding_batch)
            logger.info(f"Added {total_added_chunks} chunks to the vector database.")
