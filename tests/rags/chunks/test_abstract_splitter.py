import pytest

from rags.chunks.abstract_splitter import (
    AbstractFileSplitter,
    AbstractTextSplitter,
    FileChunk,
    RagDocument,
    TextChunk,
)


# Test RagDocument initialization
def test_rag_document_init():
    doc = RagDocument(content='abc', metadata={'a': 1})
    assert doc.content == 'abc'
    assert doc.metadata == {'a': 1}

# Test FileChunk initialization
def test_file_chunk_init():
    chunk = FileChunk(content='xyz', metadata={'b': 2})
    assert chunk.content == 'xyz'
    assert chunk.metadata == {'b': 2}

# Test TextChunk initialization
def test_text_chunk_init():
    chunk = TextChunk(content='foo', metadata={'c': 3})
    assert chunk.content == 'foo'
    assert chunk.metadata == {'c': 3}

# Test AbstractTextSplitter split_text is abstract
def test_abstract_text_splitter_split_text_abstract():
    class Dummy(AbstractTextSplitter):
        pass
    with pytest.raises(TypeError):
        Dummy()

# Test AbstractFileSplitter __init__ sets attributes and creates splitters
def test_abstract_file_splitter_init(mocker):
    mock_token = mocker.patch('rags.chunks.abstract_splitter.tiktoken.get_encoding')
    mock_token.return_value = 'encoder'
    class Dummy(AbstractFileSplitter):
        def split_file(self, input_document):
            pass
        def load_file(self):
            pass
    dummy = Dummy('file', include_token_limit=True, include_byte_limit=True)
    assert dummy.path_to_file == 'file'
    assert dummy.encoder == 'encoder'
