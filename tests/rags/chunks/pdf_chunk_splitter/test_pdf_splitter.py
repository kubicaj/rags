from unittest.mock import MagicMock

from rags.chunks.pdf_chunk_splitter.pdf_splitter import PdfFileSplitter


# Test PdfFileSplitter initialization
def test_init_sets_path_and_inits_super(mocker):
    # Test that PdfFileSplitter calls super().__init__ with correct args
    mock_super = mocker.patch('rags.chunks.pdf_chunk_splitter.pdf_splitter.AbstractFileSplitter.__init__')
    PdfFileSplitter('file.pdf')
    mock_super.assert_called_once_with('file.pdf', include_token_limit=True, include_byte_limit=True)

# Test load_file returns RagDocument with correct content and metadata
def test_load_file_reads_pdf_and_returns_ragdocument(mocker):
    mock_fitz_open = mocker.patch('rags.chunks.pdf_chunk_splitter.pdf_splitter.fitz.open')
    mock_rag_doc = mocker.patch('rags.chunks.pdf_chunk_splitter.pdf_splitter.RagDocument')
    mock_pdf = MagicMock()
    mock_fitz_open.return_value = mock_pdf
    mock_rag_doc.return_value = 'rag_doc_obj'
    splitter = PdfFileSplitter('file.pdf')
    result = splitter.load_file()
    mock_fitz_open.assert_called_once_with('file.pdf')
    mock_rag_doc.assert_called_once_with(content=mock_pdf, metadata={'source': 'file.pdf', 'type': 'pdf'})
    assert result == 'rag_doc_obj'

# Test split_file returns FileChunk list
def test_split_file_returns_chunks(mocker):
    mock_file_chunk = mocker.patch('rags.chunks.pdf_chunk_splitter.pdf_splitter.FileChunk')
    splitter = PdfFileSplitter('file.pdf')
    splitter._extract_header_chunks = MagicMock(return_value=['chunk1', 'chunk2'])
    mock_file_chunk.side_effect = lambda content, metadata: (content, metadata)
    result = splitter.split_file(MagicMock(content='pdf_content'))
    assert result == [('chunk1', {'source': 'file.pdf'}), ('chunk2', {'source': 'file.pdf'})]

# Test _extract_header_chunks static method
def test_extract_header_chunks_extracts_chunks(mocker):
    # Mocks fitz.open and simulates PDF structure
    mock_fitz_open = mocker.patch('rags.chunks.pdf_chunk_splitter.pdf_splitter.fitz.open')
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.__iter__.return_value = [mock_page]
    mock_page.get_text.return_value = {'blocks': [
        {'lines': [{'spans': [{'text': 'Header', 'size': 16}, {'text': 'Content', 'size': 12}]}]},
        {'lines': [{'spans': [{'text': 'Header2', 'size': 16}]}]}
    ]}
    mock_fitz_open.return_value = mock_doc
    result = PdfFileSplitter._extract_header_chunks('fake_path', min_header_font=14)
    assert isinstance(result, list)
    mock_doc.close.assert_called_once()

# Test split_file with no chunks
def test_split_file_empty_chunks(mocker):
    mocker.patch('rags.chunks.pdf_chunk_splitter.pdf_splitter.FileChunk')
    splitter = PdfFileSplitter('file.pdf')
    splitter._extract_header_chunks = MagicMock(return_value=[])
    result = splitter.split_file(MagicMock(content='pdf_content'))
    assert result == []

