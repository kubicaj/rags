from abc import ABC, abstractmethod


class AbstractEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text (str): The input text to embed.
        Return:
            (list[float]): The embedding vector.
        """
        pass
