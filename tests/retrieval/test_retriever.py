"""Retriever tests - in-memory store + fake embeddings: no Chroma, no OpenAI.

DeterministicFakeEmbedding gives the same vector for the same text, so a query
that is exactly a chunk's text is guaranteed to match that chunk best.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from rights_assistant_agent.domain import RetrievedDoc
from rights_assistant_agent.retrieval.retriever import KnowledgeBaseRetriever

CHUNKS = [
    (
        "severance.txt#0",
        "פיצויי פיטורים",
        "פיצויי פיטורים\n\nעובד שפוטר אחרי שנה זכאי לפיצויים",
    ),
    ("leave.txt#0", "חופשה שנתית", "חופשה שנתית\n\nכל עובד זכאי לימי חופשה בתשלום"),
    (
        "notice.txt#0",
        "הודעה מוקדמת",
        "הודעה מוקדמת\n\nמעסיק חייב לתת הודעה מוקדמת בכתב",
    ),
]


class ScoredInMemoryVectorStore(InMemoryVectorStore):
    """InMemoryVectorStore doesn't implement relevance scores; its raw score is
    already cosine similarity (higher = better), so pass it through unchanged."""

    def _select_relevance_score_fn(self):
        return lambda score: score


@pytest.fixture
def retriever() -> KnowledgeBaseRetriever:
    store = ScoredInMemoryVectorStore(DeterministicFakeEmbedding(size=64))
    store.add_documents(
        [
            Document(
                id=chunk_id,
                page_content=text,
                metadata={"title": title, "url": f"https://example.org/{chunk_id}"},
            )
            for chunk_id, title, text in CHUNKS
        ]
    )
    return KnowledgeBaseRetriever(store)


def test_returns_domain_objects_with_metadata(retriever):
    results = retriever.search(CHUNKS[1][2], k=1)

    assert len(results) == 1
    doc = results[0]
    assert isinstance(doc, RetrievedDoc)
    assert doc.title == "חופשה שנתית"
    assert doc.url == "https://example.org/leave.txt#0"
    assert doc.text == CHUNKS[1][2]


def test_best_match_comes_first(retriever):
    results = retriever.search(CHUNKS[2][2], k=3)
    assert results[0].title == "הודעה מוקדמת"


def test_results_are_sorted_by_score_best_first(retriever):
    scores = [d.score for d in retriever.search("עובד", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_respects_k(retriever):
    assert len(retriever.search("עובד", k=2)) == 2


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_query_returns_empty_list(retriever, query):
    assert retriever.search(query, k=3) == []
