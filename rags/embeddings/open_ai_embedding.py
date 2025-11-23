from openai import OpenAI

from rags import global_settings
from rags.embeddings.abstract_embedding import AbstractEmbedding


class OpenAIEmbedding(AbstractEmbedding):
    """
    OpenAI embedding implementation using the OpenAI API.
    """

    def __init__(self, api_key: str, embedding_model: str = global_settings.OPEN_AI_EMBEDDING_MODEL):
        """
        Initialize the OpenAIEmbedding with the given API key and embedding model.
        Args:
            api_key (str): The OpenAI API key.
            embedding_model (str): The OpenAI embedding model to use.
        """
        self.open_ai_client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text using OpenAI's embedding model.
        Args:
            text (str): The input text to embed.
        Return:
            (list[float]): The embedding vector.
        """
        response = self.open_ai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
