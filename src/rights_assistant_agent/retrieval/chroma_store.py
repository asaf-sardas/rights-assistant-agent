"""Open the persistent Chroma vector store.

The embedding model and distance metric are applied when the collection is
created. The embedding model name is also saved in the collection's metadata:
opening it with a different model raises an error, because vectors from
different models cannot be compared.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class EmbeddingModelMismatchError(RuntimeError):
    pass


def open_chroma_store(
    persist_dir: Path,
    collection_name: str,
    embeddings: Embeddings,
    embedding_model: str,
    distance_metric: str,
) -> Chroma:
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"embedding_model": embedding_model},
        collection_configuration={"hnsw": {"space": distance_metric}},
    )

    stored_model = (store._collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != embedding_model:
        raise EmbeddingModelMismatchError(
            f"Collection '{collection_name}' was built with '{stored_model}', "
            f"but settings use '{embedding_model}'. "
            f"Delete '{persist_dir}' and run the ingest script again."
        )
    return store
