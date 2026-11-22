"""Build the vector index from the .txt pages in data/raw/.

Usage:
    uv run python scripts/ingest.py

Rebuilds the collection from scratch on every run, so the index always
matches the files exactly (no duplicates, no chunks of deleted pages).
"""

import sys

from langchain_openai import OpenAIEmbeddings

from rights_assistant_agent.config import get_settings
from rights_assistant_agent.retrieval.chroma_store import open_chroma_store
from rights_assistant_agent.retrieval.indexer import index_chunks, split_documents
from rights_assistant_agent.retrieval.loaders.txt_loader import load_txt_documents


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Hebrew in the Windows console
    settings = get_settings()

    pages = load_txt_documents(settings.data_dir)
    print(f"Loaded {len(pages)} pages from {settings.data_dir}")

    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    store = open_chroma_store(
        settings.chroma_dir,
        settings.chroma_collection,
        embeddings,
        settings.embedding_model,
        settings.distance_metric,
    )
    store.reset_collection()  # start clean: the files are the source of truth

    ids = index_chunks(store, chunks)
    print(
        f"Indexed {len(ids)} chunks into '{settings.chroma_collection}' at {settings.chroma_dir}"
    )


if __name__ == "__main__":
    main()
