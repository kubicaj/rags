import pytest

from rags.embeddings.abstract_embedding import AbstractEmbedding


# Test that AbstractEmbedding cannot be instantiated directly
def test_abstract_embedding_instantiation():
    # Should raise TypeError because of abstract method
    with pytest.raises(TypeError):
        AbstractEmbedding()

# Test that a subclass must implement embed
def test_subclass_without_embed():
    # Dynamically create a subclass without implementing embed
    class BadEmbedding(AbstractEmbedding):
        pass

    with pytest.raises(TypeError):
        BadEmbedding()

# Test that a subclass with embed can be instantiated and called
def test_subclass_with_embed():
    class GoodEmbedding(AbstractEmbedding):
        def embed(self, text: str):
            return [1.0, 2.0]

    emb = GoodEmbedding()
    assert emb.embed("test") == [1.0, 2.0]

