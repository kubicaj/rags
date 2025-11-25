# rags: Retrieval-Augmented Generation Platform

This repository provides a modular framework for Retrieval-Augmented Generation (RAG) tasks, allowing users to
preprocess, embed, store, index, and retrieve chunks of documents efficiently using various vector store backends and
embedding models.

## Repository Structure

```
rags/
│
├── chunks/                               # Document/text chunking implementations
│   ├── abstract_splitter.py              # Base class for chunk splitters. Contains classes for Text and Files chunking.
│   ├── chunks_splitter_factory.py        # Factory to create chunk splitters based on file type
│   ├── md_chunck_splitter/               # Markdown-specific chunking
│   ├── pdf_chunk_splitter/               # PDF-specific chunking
│   └── string_splitters/                 # Generic string-based chunking
│
├── vector_database/                      # Vector storage/search abstractions & backends
│   ├── abstract_vector_database.py
│   └── (e.g., s3_vector_bucket.py)       # Example vector DB backend implementation
│   └── vector_database_factory.py        # Factory to create vector DB instances based on configuration
│
├── embeddings/                           # Embedding model integration and factories
│   └── embedding_factory.py
│   └── (e.g., open_ai_embedding.py)      # Example embedding model implementation
│   └── embedding_factory.py              # Factory to create embedding model instances
│
├── common/                               # Shared utilities
│   └── setup_logger.py
│
├── global_settings.py                    # Core configuration/constants
├── rag_driver.py                         # RAG orchestration & API
```

## Core Concepts

- **Chunks:** Documents are broken up into smaller, context-aware pieces using specialized splitters.
- **Embeddings:** Each chunk is converted to a vector representation, enabling semantic search.
- **Vector Database:** Embeddings are indexed and searched using pluggable vector databases.
- **RagDriver:** High-level API that coordinates chunking, embedding, indexing, and querying.

## Getting Started

Here's a quick guide to set up and use the RAG platform.

#### Environment Variables

Here are the required environment variables for the application:
(You can set these in `.env` or your environment)

- `OPENAI_API_KEY`: Your OpenAI API key for accessing OpenAI services.
- `AWS_ACCESS_KEY_ID`: Your AWS access key ID for accessing AWS services.
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key.
- `AWS_REGION`: The AWS region for your resources (e.g., `us-east-1`).

#### Support Tools

Here are some of the key tools and libraries used in this project:

- **UV** – Python package manager (see `uv.lock`).
- **Ruff** – Linter for code quality.
- **Pytest** – Testing framework.
- **GitHub Actions** – Workflow automation for CI.

---

## Enhancements and Extensions

Here's how you can extend the platform by adding new vector database backends, embeddings or chunk splitters.

### How to Add a New Vector Database Backend

1. **Subclass the Abstract Interface:**
   Implement your backend by subclassing the appropriate abstract class in
   `vector_database/abstract_vector_database.py`.
2. **Implement Required Methods:**
   Methods typically include adding vectors, querying for similarity, and index management (create/delete).
3. **Register with the Factory:**
   Edit `vector_database_factory.py` to expose your backend as a configuration option.
4. **Test Integration:**
   Make sure your backend works with the workflow in `RagDriver` and can be selected by users.

---

### How to Add a New ChunkSplitter

1. **Subclass AbstractFileSplitter:**
   Implement a new splitter under `chunks/` (or its subfolders), inheriting `AbstractFileSplitter`.
2. **Provide Chunking Logic:**
   Core method: `create_chunks(self) -> list[FileChunk]` according to your splitting rules.
3. **Register with the Factory:**
   Update `chunks_splitter_factory.py` for your splitter to be used based on file type/extension.
4. **Test on Sample Data:**
   Ensure your splitter works by running with representative documents.

---

### How to Add a New Embedding Model

1. **Subclass AbstractEmbedding:**
   Implement a new embedding under `embeddings/` (or its subfolders), inheriting `AbstractEmbedding`.
2. **Provide Chunking Logic:**
   Core method: `embed(self, text: str) -> list[float]:` with your custom logic of creation of embeddings.
3. **Register with the Factory:**
   Update `embedding_factory.py` for your splitter to be used based on file type/extension.
4. **Test on Sample Data:**
   Ensure your splitter works by running with representative documents.

---

## Example of Usage

Before do not forget to setup environment variables

1. (optional) Update settings within the module `global_settings.py`

2. Create instance of `RagDriver`
   ```python
   # Sample with default config
   rag_driver = RagDriver(embedding_type="openai")
   
   # Sample with S3 vector DB configuration
   rag_driver = RagDriver(
      embedding_type="openai",
      vector_database_options={
          "s3_vector_db_config": S3VectorBucketConfig(
              bucket_name="your-bucket-name",
              index_name="your-index-name",
              dataType="float32",
              dimension=3072,  # this should match the embedding dimension used
              distance_metric="cosine",
              non_filterable_metadata_keys=["content", "source"]
          )
   )
   ```
3. Use `RagDriver` to fill your vector database:
   ```python
   rag_driver.fill_rag(source_path="/path/to/data/")
   ```
4. Query relevant content:
   ```python
   results = rag_driver.find_in_rag("your query", top_k=5)
   for result in results:
       print(result)
   ```
---

## Possible Enhancements

- Add batch processing of embeddings for improved performance. (
  See [OpenAI batch docs](https://platform.openai.com/docs/guides/batch))
- Add richer settings management for configuration.
