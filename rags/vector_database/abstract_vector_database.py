from abc import ABC, abstractmethod


class AbstractVectorDatabase(ABC):
    """
    Abstract base class for vector database implementations.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def add_vectors(self, vectors: list[list[float]], metadata: list[dict]):
        """
        Add vectors to the vector database.

        Args:
            vectors (list[list[float]]): List of vectors to add.
            metadata (list[dict]): List of metadata corresponding to each vector.
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
