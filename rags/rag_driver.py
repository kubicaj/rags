import logging
import os
from typing import List, Literal

from rags.chunks.abstract_splitter import AbstractFileSplitter, FileChunk
from rags.chunks.chunks_splitter_factory import ChunkSplitterFactory
from rags.embeddings.embedding_factory import EmbeddingFactory


class RagDriver:

    def __init__(self, embedding_type: Literal["openai"] = "openai"):
        """
        Initialize the RAG driver with the specified embedding type.

        Args:
            embedding_type (Literal["openai"]): The embedding implementation to use. Default is "openai".
        """
        # create basic logger
        logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
        # init other components
        self.logger = logging.getLogger(__name__)
        self.embedding_instance = EmbeddingFactory.create(embedding_type)



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
            self.logger.warning("File type %s is not supported and will be skipped.", file_type)

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
        result_chunks: List[FileChunk] = []
        for file_path in file_to_process:
            self.logger.info("Processing file: %s", file_path)
            splitter: AbstractFileSplitter = ChunkSplitterFactory.create_based_on_file_type(file_path)
            result_chunks.extend(splitter.create_chunks())

        # ------------------------------------------------
        # Step 2 - generate embeddings
        # ------------------------------------------------
        self.logger.info("Total chunks created: %d", len(result_chunks))
        for chunk in result_chunks:
            embedding = self.embedding_instance.embed(chunk.content)
            print(embedding)

        # TODO

RagDriver().fill_rag(r"D:\Repositories\rags\resources\large_sample.md")
