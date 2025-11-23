import fitz
import pymupdf

from rags.chunks.abstract_splitter import AbstractFileSplitter, FileChunk, RagDocument


class PdfFileSplitter(AbstractFileSplitter):
    """
    A PDF file splitter that splits PDF documents into chunks based on headers.

    Sample Usage:
    ```python
    pdf_splitter = PdfFileSplitter("path/to/pdf/file.pdf")
    pdf_splitter.create_chunks()
    ```
    """

    def __init__(self, path_to_pdf_file: str):
        """
        Initialize the PdfFileSplitter with the path to the PDF file.

        Args:
            path_to_pdf_file (str): Path to the PDF file to be split.
        """
        super().__init__(path_to_pdf_file, include_token_limit=True, include_byte_limit=True)

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
        pdf_doc = fitz.open(self.path_to_file)
        return RagDocument(
            content=pdf_doc,
            metadata={
                "source": self.path_to_file,
                "type": "pdf"
            }
        )

    def split_file(self, input_document) -> list[FileChunk]:
        """
        [Implementation of AbstractSplitter]
        Split the PDF document into chunks based on headers.

        Arg:
            input_document (RagDocument): Content of the PDF file to be split.
        Return:
            (list[FileChunk]): List of file chunks with metadata.
        """
        chunks = self._extract_header_chunks(input_document.content)
        return [FileChunk(content=chunk, metadata={"source": self.path_to_file}) for chunk in chunks]

    @staticmethod
    def _extract_header_chunks(pdf_path: pymupdf.Document, min_header_font=14) -> list[str]:
        """
        Extract text chunks from a PDF based on header font sizes.

        Args:
            pdf_path (str): Path to the PDF file.
            min_header_font (int): Minimum font size to consider as a header.

        Return:
            (list[str]): List of text chunks split by headers.
        """
        doc = fitz.open(pdf_path)
        chunks = []
        current = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]

            for b in blocks:
                if "lines" not in b:
                    continue

                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        if span["size"] >= min_header_font:
                            if current:
                                chunks.append("\n".join(current))
                                current = []
                        current.append(text)

        if current:
            chunks.append("\n".join(current))

        doc.close()
        return chunks
