
import pytest

from rags.chunks.chunks_splitter_factory import ChunkSplitterFactory


# Test PDF file type returns PdfFileSplitter
def test_create_based_on_file_type_pdf(mocker):
    mock_pdf_splitter = mocker.patch('rags.chunks.chunks_splitter_factory.PdfFileSplitter')
    result = ChunkSplitterFactory.create_based_on_file_type('file.pdf')
    mock_pdf_splitter.assert_called_once_with('file.pdf')
    assert result == mock_pdf_splitter.return_value

# Test MD file type returns LangChainMDFileSplitter
def test_create_based_on_file_type_md(mocker):
    mock_md_splitter = mocker.patch('rags.chunks.chunks_splitter_factory.LangChainMDFileSplitter')
    result = ChunkSplitterFactory.create_based_on_file_type('file.md', md_file_encoding='utf-16')
    mock_md_splitter.assert_called_once_with('file.md', md_file_encoding='utf-16')
    assert result == mock_md_splitter.return_value

# Test unsupported file type raises ValueError
def test_create_based_on_file_type_unsupported():
    with pytest.raises(ValueError):
        ChunkSplitterFactory.create_based_on_file_type('file.txt')

