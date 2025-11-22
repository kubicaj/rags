from langchain_text_splitters import MarkdownHeaderTextSplitter

from rags.chunks.abstract_splitter import AbstractFileSplitter, RagDocument, FileChunk


class LangChainMDFileSplitter(AbstractFileSplitter):
    """
    A MD file splitter that splits MD documents into chunks based on Markdown headers using LangChain's

    Sample Usage:
    ```python
    md_splitter = LangChainMDFileSplitter("path/to/md/file.md")
    md_splitter.create_chunks()
    ```
    """

    HEADERS_TO_SPLIT_ON = [
        ("#", "Doc Section level 1"),
        ("##", "Doc Section level 2"),
        ("###", "Doc Section level 3"),
    ]

    def __init__(self, path_to_md_file: str, md_file_encoding: str = "utf-8"):
        """
        Initialize the MdFileSplitter with the path to the MD file.

        Args:
            path_to_md_file (str): Path to the MD file to be split.
        """
        super().__init__(path_to_md_file, include_token_limit=True, include_byte_limit=True)
        # Markdown header-based splitter
        self.md_file_encoding = md_file_encoding
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
            return_each_line=False,
            strip_headers=False,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Abstract Methods Implementation
    # ------------------------------------------------------------------------------------------------------------------

    def load_file(self) -> RagDocument:
        """
        [Implementation of AbstractSplitter]
        Load the content of the PDF file.

        Return:
            (RagDocument): Loaded PDF document
        """
        with open(self.path_to_file, "r", encoding=self.md_file_encoding, errors="ignore") as f:
            content = f.read()
            return RagDocument(
                content=content,
                metadata={
                    "source": self.path_to_file,
                    "type": "MD"
                }
            )

    def split_file(self, input_document: RagDocument) -> list[FileChunk]:
        """
        [Implementation of AbstractSplitter]
        Split the PDF document into chunks based on headers.

        Arg:
            input_document (RagDocument): Content of the PDF file to be split.
        Return:
            (list[FileChunk]): List of file chunks with metadata.
        """
        doc_chunks = self.splitter.split_text(input_document.content)

        result_chunks: list[FileChunk] = []
        for doc_chunk in doc_chunks:
            metadata = doc_chunk.metadata
            metadata["source"] = self.path_to_file
            result_chunks.append(
                FileChunk(
                    content=f"{doc_chunk.metadata}\n\n{doc_chunk.page_content}",
                    metadata=metadata
                )
            )
        return result_chunks
