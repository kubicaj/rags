from rags.chunks.string_splitters.bytes_text_splitter import BytesSplitter

# Test BytesSplitter initialization

def test_init_sets_chunk_size_and_overlap():
    # Test that BytesSplitter initializes with correct chunk_size and chunk_overlap
    splitter = BytesSplitter(chunk_size=100, chunk_overlap=10)
    assert splitter.chunk_size == 100
    assert splitter.chunk_overlap == 10

# Test split_text returns correct TextChunk list

def test_split_text_returns_chunks(mocker):
    # Test that split_text returns a list of TextChunk with correct content and metadata
    mock_text_chunk = mocker.patch('rags.chunks.string_splitters.bytes_text_splitter.TextChunk')
    splitter = BytesSplitter(chunk_size=4, chunk_overlap=2)
    # 'abcdef' encoded as utf-8 is 6 bytes
    splitter.split_text('abcdef')
    # Should create chunks: 'abcd', 'cdef', 'ef'
    assert mock_text_chunk.call_count == 3
    expected = [
        ('abcd', {'chunk_num': 0}),
        ('cdef', {'chunk_num': 1}),
        ('ef', {'chunk_num': 2})
    ]
    for call, (exp_content, exp_metadata) in zip(mock_text_chunk.call_args_list, expected):
        assert call.kwargs['content'] == exp_content
        assert call.kwargs['metadata'] == exp_metadata

# Test split_text with empty input

def test_split_text_empty(mocker):
    # Test that split_text returns an empty list when input_text is empty
    mocker.patch('rags.chunks.string_splitters.bytes_text_splitter.TextChunk')
    splitter = BytesSplitter(chunk_size=4, chunk_overlap=2)
    result = splitter.split_text('')
    assert result == []
