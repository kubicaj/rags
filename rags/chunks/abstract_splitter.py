from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, List

import tiktoken
from loguru import logger

from rags import global_settings


class RagDocument:
    """
    A class representing a document with its content and metadata.
    """

    def __init__(self, content: Any, metadata: dict):
        self.content = content
        self.metadata = metadata


class FileChunk:
    """
    A class representing a chunk of a file with its content and metadata.
    """

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata


class TextChunk:
    """
    A class representing a chunk of a text with its content and metadata.
    """

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata


class AbstractTextSplitter(ABC):
    """
    Abstract base class for text splitters.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def split_text(self, input_text: str) -> List[TextChunk]:
        """
        Split the text into chunks.

        Arg:
            text (str): The text to be split.
            input_text (str): The text to be split.
        Return:
            (list[str]): List of text chunks.
        """
        pass


class AbstractFileSplitter(ABC):
    """
    Abstract base class for file splitters.
    """

    NUM_TOKENS_KEY = 'num_tokens'
    NUM_BYTES_KEY = 'num_bytes'

    def __init__(self, path_to_file: str, include_token_limit: bool = True, include_byte_limit: bool = True):
        """
        Initialize the AbstractSplitter.

        Args:
            path_to_file (str): Path to the file to be split.
            include_token_limit (bool): Whether to include token limit from global config_files.
            include_byte_limit (bool): Whether to include byte limit from global config_files.
        """
        self.path_to_file = path_to_file
        self.encoder = tiktoken.get_encoding(global_settings.TOKENIZER_NAME)

        self.token_limit = global_settings.EMBEDDING_MODEL_TOKENS_LIMIT
        self.byte_limit = global_settings.S3_VECTOR_INDEX_METADATA_BYTES_LIMIT

        self.token_splitter = None
        if include_token_limit:
            from rags.chunks.string_splitters.token_text_splitter import TokenSplitter
            self.token_splitter = TokenSplitter(
                chunk_size=self.token_limit - 2000,  # tokens limit minus twice for overlap
                chunk_overlap=1000
            )
        self.bytes_splitter = None
        if include_byte_limit:
            from rags.chunks.string_splitters.bytes_text_splitter import BytesSplitter
            self.bytes_splitter = BytesSplitter(
                chunk_size=self.byte_limit - 1024 - 2000,
                # reserve some bytes for metadata minus twice for overlap
                chunk_overlap=1000
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def split_file(self, input_document: RagDocument) -> List[FileChunk]:
        """
        Split the file content into chunks.

        Arg:
            input_document (Document): Content of the file to be split.
        Return:
            (list[FileChunk]): List of file chunks with metadata.
        """
        pass

    @abstractmethod
    def load_file(self) -> RagDocument:
        """
        Load the content of the file.

        Return:
            (RagDocument): Loaded file
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Static Methods
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def count_bytes(text: str) -> int:
        """
        Count the number of bytes in a text string.

        Args:
            text (str): The text string to measure.

        Return:
            (int): The number of bytes in the string.
        """
        return len(text.encode("utf-8"))

    # ------------------------------------------------------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------------------------------------------------------

    def _filter_by_bytes(self, file_chunks: List[FileChunk]) -> List[FileChunk]:
        """
        Filter and further split chunks that exceed the byte limit.

        Args:
            file_chunks (List[FileChunk]): List of file chunks to be filtered.
        Return:
            (List[FileChunk]): List of filtered file chunks.
        """
        if not self.bytes_splitter:
            return file_chunks

        filtered_chunks = []
        for chunk in file_chunks:
            if chunk.metadata[self.NUM_BYTES_KEY] <= self.byte_limit:
                filtered_chunks.append(chunk)
            else:
                # Split the chunk further using the bytes splitter
                sub_chunks = self.bytes_splitter.split_text(chunk.content)
                new_file_chunks = [
                    FileChunk(content=sub_chunk.content, metadata=deepcopy(chunk.metadata)) for sub_chunk in sub_chunks
                ]
                # recalculate metadata for new chunks
                self._calculate_metadata_statistics(new_file_chunks)
                filtered_chunks.extend(new_file_chunks)
        return filtered_chunks

    def _filter_by_tokens(self, file_chunks: List[FileChunk]) -> List[FileChunk]:
        """
        Filter and further split chunks that exceed the token limit.

        Args:
            file_chunks (List[FileChunk]): List of file chunks to be filtered.
        Return:
            (List[FileChunk]): List of filtered file chunks.
        """
        if not self.token_splitter:
            return file_chunks

        filtered_chunks = []
        for chunk in file_chunks:
            if chunk.metadata[self.NUM_TOKENS_KEY] <= self.token_limit:
                filtered_chunks.append(chunk)
            else:
                # Split the chunk further using the token splitter
                sub_chunks = self.token_splitter.split_text(chunk.content)
                new_file_chunks = [
                    FileChunk(content=sub_chunk.content, metadata=deepcopy(chunk.metadata)) for sub_chunk in sub_chunks
                ]
                # recalculate metadata for new chunks
                self._calculate_metadata_statistics(new_file_chunks)
                filtered_chunks.extend(new_file_chunks)
        return filtered_chunks

    def _calculate_metadata_statistics(self, file_chunks: List[FileChunk]) -> List[FileChunk]:
        """
        Calculate and update metadata statistics for each file chunk.

        Args:
            file_chunks (List[FileChunk]): List of file chunks to analyze.
        Return:
            (List[FileChunk]): List of file chunks with updated metadata statistics.
        """
        for chunk in file_chunks:
            chunk.metadata[self.NUM_TOKENS_KEY] = self.count_tokens(chunk.content)
            chunk.metadata[self.NUM_BYTES_KEY] = self.count_bytes(chunk.content)
        return file_chunks

    # ------------------------------------------------------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text (str): The text string to encode.

        Return:
            (int): The number of tokens in the string.
        """
        return len(self.encoder.encode(text))

    def create_chunks(self) -> List[FileChunk]:
        """
        Load and split the file into chunks.

        Return:
            (list[FileChunk]): List of file chunks with metadata.
        """
        document = self.load_file()
        list_of_chunks = self.split_file(document)
        if not list_of_chunks:
            raise ValueError(f"No chunks were created from the file: {self.path_to_file}")

        self._calculate_metadata_statistics(list_of_chunks)
        list_of_chunks = self._filter_by_bytes(list_of_chunks)
        list_of_chunks = self._filter_by_tokens(list_of_chunks)

        # per each chunk print its metadata
        for chunk in list_of_chunks:
            logger.debug(f"Chunk metadata: {chunk.metadata}")

        # calculate total chunks tokens and bytes
        total_tokens = sum(chunk.metadata[self.NUM_TOKENS_KEY] for chunk in list_of_chunks)
        total_bytes = sum(chunk.metadata[self.NUM_BYTES_KEY] for chunk in list_of_chunks)
        logger.info(
            f"\n"
            f"====================================================================\n"
            f"Summary for file: {self.path_to_file} \n"
            f"    Total chunks created for file: {len(list_of_chunks)}, \n"
            f"    Total tokens: {total_tokens}, \n"
            f"    Total bytes: {total_bytes}\n"
            f"===================================================================="
            f"\n")
        return list_of_chunks
