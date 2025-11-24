from unittest.mock import MagicMock, patch

from rags.chunks.string_splitters.token_text_splitter import TokenSplitter

# Test TokenSplitter initialization

def test_init_sets_token_splitter():
    # Test that TokenSplitter initializes with correct parameters
    with patch('rags.chunks.string_splitters.token_text_splitter.TokenTextSplitter') as mock_token_splitter:
        splitter = TokenSplitter(chunk_size=123, chunk_overlap=45, tokenizer_name='test-enc')
        mock_token_splitter.assert_called_once_with(encoding_name='test-enc', chunk_size=123, chunk_overlap=45)
        assert hasattr(splitter, 'token_splitter')

# Test split_text returns correct TextChunk list

def test_split_text_returns_chunks(mocker):
    # Test that split_text returns a list of TextChunk with correct content and metadata
    mocker.patch('rags.chunks.string_splitters.token_text_splitter.TokenTextSplitter')
    mock_text_chunk = mocker.patch('rags.chunks.string_splitters.token_text_splitter.TextChunk')
    splitter = TokenSplitter()
    splitter.token_splitter = MagicMock()
    splitter.token_splitter.split_text.return_value = ['chunk1', 'chunk2']
    mock_text_chunk.side_effect = lambda content, metadata: (content, metadata)
    result = splitter.split_text('some text')
    assert result == [('chunk1', {'chunk_num': 0}), ('chunk2', {'chunk_num': 1})]
    splitter.token_splitter.split_text.assert_called_once_with('some text')

# Test split_text with empty input

def test_split_text_empty(mocker):
    # Test that split_text returns an empty list when input_text is empty
    mocker.patch('rags.chunks.string_splitters.token_text_splitter.TokenTextSplitter')
    mocker.patch('rags.chunks.string_splitters.token_text_splitter.TextChunk')
    splitter = TokenSplitter()
    splitter.token_splitter = MagicMock()
    splitter.token_splitter.split_text.return_value = []
    result = splitter.split_text('')
    assert result == []

