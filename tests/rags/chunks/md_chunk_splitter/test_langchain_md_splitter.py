import pytest

from rags.chunks.md_chunck_splitter.langchain_md_splitter import LangChainMDFileSplitter


@pytest.fixture
def mock_rag_document(mocker):
    # Fixture to provide a mock RagDocument with sample content and metadata
    mock_doc = mocker.Mock()
    mock_doc.content = "Header1\nContent1\n## Header2\nContent2"
    mock_doc.metadata = {"source": "fake_path", "type": "MD"}
    return mock_doc

def test_init_sets_attributes(mocker):
    # Test that the splitter initializes attributes and uses the correct splitter instance
    mock_splitter_cls = mocker.patch(
        "rags.chunks.md_chunck_splitter.langchain_md_splitter.MarkdownHeaderTextSplitter"
    )
    mock_instance = mocker.Mock()
    mock_splitter_cls.return_value = mock_instance
    splitter = LangChainMDFileSplitter("some/path.md", md_file_encoding="ascii")
    assert splitter.path_to_file == "some/path.md"
    assert splitter.md_file_encoding == "ascii"
    assert splitter.splitter is mock_instance

def test_load_file_reads_file_and_returns_ragdocument(mocker):
    # Test that load_file reads the file and returns a RagDocument with correct content and metadata
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data="file content"))
    mock_rag_doc_cls = mocker.patch(
        "rags.chunks.md_chunck_splitter.langchain_md_splitter.RagDocument"
    )
    splitter = LangChainMDFileSplitter("test.md")
    mock_rag_doc_cls.return_value = "mocked_rag_doc"
    result = splitter.load_file()
    mock_open.assert_called_once_with("test.md", "r", encoding="utf-8", errors="ignore")
    mock_rag_doc_cls.assert_called_once_with(
        content="file content",
        metadata={"source": "test.md", "type": "MD"}
    )
    assert result == "mocked_rag_doc"

def test_split_file_splits_and_returns_chunks(mocker, mock_rag_document):
    # Test that split_file splits the document and returns chunks with updated metadata
    mock_file_chunk = mocker.patch(
        "rags.chunks.md_chunck_splitter.langchain_md_splitter.FileChunk"
    )
    splitter = LangChainMDFileSplitter("file.md")
    splitter.splitter = mocker.Mock()
    doc_chunk1 = mocker.Mock()
    doc_chunk1.metadata = {"header": "Header1"}
    doc_chunk1.page_content = "Content1"
    doc_chunk2 = mocker.Mock()
    doc_chunk2.metadata = {"header": "Header2"}
    doc_chunk2.page_content = "Content2"
    splitter.splitter.split_text.return_value = [doc_chunk1, doc_chunk2]
    mock_file_chunk.side_effect = lambda content, metadata: (content, metadata)
    result = splitter.split_file(mock_rag_document)
    assert len(result) == 2
    assert result[0][0].startswith("{'header': 'Header1'")
    assert result[0][1]["source"] == "file.md"
    assert result[1][0].startswith("{'header': 'Header2'")
    assert result[1][1]["source"] == "file.md"

def test_split_file_empty_chunks(mocker, mock_rag_document):
    # Test that split_file returns an empty list when there are no chunks
    mocker.patch("rags.chunks.md_chunck_splitter.langchain_md_splitter.FileChunk")
    splitter = LangChainMDFileSplitter("file.md")
    splitter.splitter = mocker.Mock()
    splitter.splitter.split_text.return_value = []
    result = splitter.split_file(mock_rag_document)
    assert result == []

def test_load_file_empty_file(mocker):
    # Test that load_file handles an empty file correctly
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data=""))
    mock_rag_doc_cls = mocker.patch(
        "rags.chunks.md_chunck_splitter.langchain_md_splitter.RagDocument"
    )
    splitter = LangChainMDFileSplitter("empty.md")
    mock_rag_doc_cls.return_value = "empty_doc"
    result = splitter.load_file()
    mock_open.assert_called_once_with("empty.md", "r", encoding="utf-8", errors="ignore")
    mock_rag_doc_cls.assert_called_once_with(
        content="",
        metadata={"source": "empty.md", "type": "MD"}
    )
    assert result == "empty_doc"

def test_load_file_file_not_found(mocker):
    # Test that load_file raises FileNotFoundError if the file does not exist
    mocker.patch("builtins.open", side_effect=FileNotFoundError)
    splitter = LangChainMDFileSplitter("missing.md")
    with pytest.raises(FileNotFoundError):
        splitter.load_file()
