import uuid
from abc import ABC, abstractmethod

from rags.chunks.abstract_splitter import FileChunk


class VectorItem:
    """
    A class representing an embedding item with its vector and metadata.
    """

    def __init__(self, key: str, embedding_vector: list[float], metadata: dict):
        """
        Initialize the VectorItem with the given embedding vector and metadata.

        Args:
            key (str): The unique key for the vector item.
            embedding_vector (list[float]): The embedding vector.
            metadata (dict): The metadata associated with the embedding vector.
        """
        self.key = key
        self.embedding_vector = embedding_vector
        self.metadata = metadata

    @staticmethod
    def create_from_file_chunk(file_chunk: FileChunk, chunk_embedding: list[float]) -> 'VectorItem':
        """
        Create a VectorItem instance from a FileChunk.

        Args:
            file_chunk (FileChunk): The FileChunk instance containing content and metadata.
            chunk_embedding (list[float]): The embedding vector for the chunk.

        Returns:
            VectorItem: A new VectorItem instance.
        """
        return VectorItem(
            key=uuid.uuid4().hex,
            embedding_vector=chunk_embedding,
            metadata={
                **file_chunk.metadata,
                "content": file_chunk.content
            }
        )

class AbstractVectorDatabase(ABC):
    """
    Abstract base class for vector database implementations.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def add_vectors(self, vectors: list[VectorItem]):
        """
        Add vectors to the vector database.

        Args:
            vectors (list[VectorItem]): A list of VectorItem instances to add to the database.
        """
        pass

    @abstractmethod
    def query_vectors(self, query_vector: list[float], top_k: int) -> list[dict]:
        """
        Query the vector database for similar vectors.

        Args:
            query_vector (list[float]): The vector to query.
            top_k (int): The number of top similar vectors to return."""

    @abstractmethod
    def create_index(self):
        """
        Create the vector index in the database.
        """
        pass

    @abstractmethod
    def delete_index(self):
        """
        Delete the vector index from the database.
        """
        pass
