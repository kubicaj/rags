# ----------------------------------------------------------------------------------------------------------------------
# General vector database settings
# ----------------------------------------------------------------------------------------------------------------------

# this is the batch size for vector upserts
BATCH_VECTOR_UPSERT_SIZE = 40

# ----------------------------------------------------------------------------------------------------------------------
# Embedding model settings
# ----------------------------------------------------------------------------------------------------------------------

# this is limit for embedding model input defined in global config_files under name TOKENIZER_NAME (see below)
EMBEDDING_MODEL_TOKENS_LIMIT = 8192
# this is the tokenizer name used for tokenizing input to embedding model
TOKENIZER_NAME = "cl100k_base"
# this is the default embedding model for openai embeddings
OPEN_AI_EMBEDDING_MODEL = "text-embedding-3-large"

# ----------------------------------------------------------------------------------------------------------------------
# S3 vector metadata settings
# ----------------------------------------------------------------------------------------------------------------------

# this is base limit for s3 vector bucket index metadata
S3_VECTOR_INDEX_METADATA_BYTES_LIMIT = 40960

# ----------------------------------------------------------------------------------------------------------------------
# Token splitter settings
# ----------------------------------------------------------------------------------------------------------------------

TOKEN_SPLITTER_DEFAULT_CHUNK_SIZE = 5000
TOKEN_SPLITTER_DEFAULT_CHUNK_OVERLAP = 1000

# ----------------------------------------------------------------------------------------------------------------------
# Bytes splitter settings
# ----------------------------------------------------------------------------------------------------------------------

BYTES_SPLITTER_DEFAULT_CHUNK_SIZE = 30000
BYTES_SPLITTER_DEFAULT_OVERLAP = 5000
