"""Indexer tests - in-memory store + fake embeddings: no Chroma, no OpenAI."""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from rights_assistant_agent.retrieval.indexer import index_chunks, split_documents


def make_page(source: str, title: str, body: str) -> Document:
    return Document(
        page_content=body,
        metadata={
            "title": title,
            "url": f"https://example.org/{source}",
            "source": source,
        },
    )


LONG_BODY = "\n\n".join(f"פסקה מספר {i}. " + "מילה " * 30 for i in range(6))


def test_chunks_keep_metadata_and_get_stable_ids():
    page = make_page("a.txt", "כותרת", LONG_BODY)

    chunks = split_documents([page], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert [c.id for c in chunks] == [f"a.txt#{i}" for i in range(len(chunks))]
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["title"] == "כותרת"
        assert chunk.metadata["url"] == "https://example.org/a.txt"
        assert chunk.metadata["chunk_index"] == i


def test_every_chunk_starts_with_page_title():
    page = make_page("a.txt", "פיצויי פיטורים", LONG_BODY)
    chunks = split_documents([page], chunk_size=200, chunk_overlap=20)
    assert all(c.page_content.startswith("פיצויי פיטורים\n\n") for c in chunks)


def test_splitting_does_not_change_the_original_page():
    page = make_page("a.txt", "כותרת", LONG_BODY)
    split_documents([page], chunk_size=200, chunk_overlap=20)
    assert page.metadata == {
        "title": "כותרת",
        "url": "https://example.org/a.txt",
        "source": "a.txt",
    }


def test_indexing_twice_does_not_create_duplicates():
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    chunks = split_documents(
        [make_page("a.txt", "A", LONG_BODY), make_page("b.txt", "B", "short body")],
        chunk_size=200,
        chunk_overlap=20,
    )

    index_chunks(store, chunks)
    index_chunks(store, chunks)

    assert len(store.store) == len(chunks)


def test_index_empty_list_is_a_no_op():
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    assert index_chunks(store, []) == []
