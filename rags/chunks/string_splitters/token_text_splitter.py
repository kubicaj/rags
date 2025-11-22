from typing import List

from rags.chunks import global_settings
from rags.chunks.abstract_splitter import AbstractTextSplitter, TextChunk

from langchain_text_splitters import TokenTextSplitter

# only defaults are defined here, actual values can be passed during initialization or taken from global settings
DEFAULT_OVERLAP = 1000
DEFAULT_CHUNK_SIZE = 5000


class TokenSplitter(AbstractTextSplitter):
    """
    A text splitter that splits text into chunks based on token limits using LangChain's TokenTextSplitter.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_SIZE,
                 tokenizer_name: str = global_settings.TOKENIZER_NAME):
        """
        Initialize the TokenSplitter with chunk size, overlap, and tokenizer name.

        Args:
            chunk_size (int): The maximum number of tokens per chunk.
            chunk_overlap (int): The number of overlapping tokens between chunks.
            tokenizer_name (str): The name of the tokenizer to use.
        """
        # Token-based splitter for further splitting large chunks
        self.token_splitter = TokenTextSplitter(
            encoding_name=tokenizer_name,  # tokenizer name
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, input_text: str) -> List[TextChunk]:
        """
        [Implementation of AbstractSplitter]
        Split the input text into chunks based on token limits.

        Args:
            input_text (str): The text to be split.

        Return:
            (list[TextChunk]): List of text chunks with metadata.
        """
        token_chunks = self.token_splitter.split_text(input_text)
        return [TextChunk(content=chunk, metadata={"chunk_num": chunk_num}) for chunk_num, chunk in
                enumerate(token_chunks)]
