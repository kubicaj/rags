import os
from typing import Literal

from dotenv import load_dotenv

from rags.embeddings.open_ai_embedding import OpenAIEmbedding

load_dotenv()


class EmbeddingFactory:
    """
    Factory class to create embedding instances based on the specified type.
    """

    @staticmethod
    def create(embedding_type: Literal["openai"], **kwargs):
        """
        Factory method to create an embedding instance based on the embedding type.

        Args:
            embedding_type (Literal["openai"]): The type of embedding to create. For now, only "openai" is supported.
            kwargs : Additional keyword arguments for the embedding instance. Possible args for OpenAIEmbedding:
                - api_key (str): The OpenAI API key. If not provided, it will be loaded from the OPENAI_API_KEY
                  environment variable.

        Return:
            (AbstractEmbedding): An instance of the specified embedding type.
        """

        if embedding_type == "openai":
            # load OPENAI_API_KEY from environment variables if not provided in kwargs
            if 'open_ai_api_key' not in kwargs:
                kwargs['api_key'] = os.getenv('OPENAI_API_KEY')
            return OpenAIEmbedding(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
