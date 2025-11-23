from rags.chunks.abstract_splitter import AbstractFileSplitter
from rags.chunks.md_chunck_splitter.langchain_md_splitter import LangChainMDFileSplitter
from rags.chunks.pdf_chunk_splitter.pdf_splitter import PdfFileSplitter


class ChunkSplitterFactory:

    @staticmethod
    def create_based_on_file_type(path_to_file: str, **kwargs) -> AbstractFileSplitter:
        """
        Factory method to create a file splitter based on the file type.

        Args:
            path_to_file (str): Path to file to be split.
            kwargs : Additional keyword arguments for the file splitter. For now, only used for MD splitter.

        Return:
            (AbstractFileSplitter): An instance of a file splitter for the specified file type.
        """
        file_type = path_to_file.split('.')[-1]
        if file_type.lower().rstrip() == 'pdf':
            return PdfFileSplitter(path_to_file)
        elif file_type.lower().rstrip() == 'md':
            return LangChainMDFileSplitter(path_to_file, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
