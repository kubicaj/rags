# this is limit for embedding model input defined in global settings under name TOKENIZER_NAME (see below)
TOKENS_LIMIT = 8192
# this is base limit for s3 vector bucket index metadata
BYTES_LIMIT = 40960
TOKENIZER_NAME = "cl100k_base"
# this is the default embedding model for openai embeddings
OPEN_AI_EMBEDDING_MODEL = "text-embedding-3-large"


def set_token_limit(limit: int = 8192):
    """
    Set the global token limit for chunking.
    Args:
        limit (int): The maximum number of tokens allowed per chunk.
    """
    global TOKENS_LIMIT
    TOKENS_LIMIT = limit


def set_bytes_limit(limit: int = 40960):
    """
    Set the global byte limit for chunking.
    Args:
        limit (int): The maximum number of bytes allowed per chunk.
    """
    global BYTES_LIMIT
    BYTES_LIMIT = limit


def set_tokenizer_name(name: str = "cl100k_base"):
    """
    Set the global tokenizer name for chunking.
    Args:
        name (str): The name of the tokenizer to use.
    """
    global TOKENIZER_NAME
    TOKENIZER_NAME = name


def set_open_ai_embedding_model(model_name: str = "text-embedding-3-large"):
    """
    Set the global OpenAI embedding model name.
    Args:
        model_name (str): The name of the OpenAI embedding model to use.
    """
    global OPEN_AI_EMBEDDING_MODEL
    OPEN_AI_EMBEDDING_MODEL = model_name
