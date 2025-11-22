from typing import List

from rags.chunks.abstract_splitter import AbstractTextSplitter, TextChunk

# only defaults are defined here, actual values can be passed during initialization or taken from global settings
DEFAULT_OVERLAP = 1000
DEFAULT_CHUNK_SIZE = 5000


class BytesSplitter(AbstractTextSplitter):
    """
    A text splitter that splits text into chunks based on token limits using LangChain's TokenTextSplitter.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_OVERLAP):
        """
        Initialize the TokenSplitter with chunk size, overlap, and tokenizer name.

        Args:
            chunk_size (int): The maximum number of bytes per chunk.
            chunk_overlap (int): The number of overlapping bytes between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, input_text: str) -> List[TextChunk]:
        """
        [Implementation of AbstractSplitter]
        Split the input text into chunks based on bytes limits.

        Args:
            input_text (str): The text to be split.

        Return:
            (list[TextChunk]): List of text chunks with metadata.
        """

        data = input_text.encode("utf-8")
        step = self.chunk_size - self.chunk_overlap
        text_chunks = []

        for start in range(0, len(data), step):
            end = start + self.chunk_size
            chunk = data[start:end].decode("utf-8", errors="replace")
            text_chunks.append(chunk)

        return [TextChunk(content=chunk, metadata={"chunk_num": chunk_num}) for chunk_num, chunk in
                enumerate(text_chunks)]
