"""Search the knowledge base and return domain objects.

Implements the `Retriever` protocol from `domain/ports.py` on top of any
LangChain VectorStore (Chroma in production, in-memory in tests).
This is the boundary where LangChain `Document`s become our `RetrievedDoc`s,
so the graph never deals with LangChain types.
"""

from langchain_core.vectorstores import VectorStore

from rights_assistant_agent.domain import RetrievedDoc


class KnowledgeBaseRetriever:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        """Return up to `k` chunks most relevant to `query`, best first.

        Rules:
        - A blank query (empty or only whitespace) returns [] without searching.
        - Use the store's `similarity_search_with_relevance_scores(query, k=k)`.
          It returns a list of (Document, score) pairs, best first,
          where score is between 0 and 1 (higher = more relevant).
        - Convert each pair into a RetrievedDoc:
            title -> from doc.metadata["title"]
            url   -> from doc.metadata["url"]
            text  -> doc.page_content
            score -> the score
        """
        if not query.strip():
            return []
        search_results = self._store.similarity_search_with_relevance_scores(
            query=query, k=k
        )
        retrieved_docs = []
        for doc, score in search_results:
            retrieved_doc = RetrievedDoc(
                title=doc.metadata["title"],
                url=doc.metadata["url"],
                text=doc.page_content,
                score=score,
            )
            retrieved_docs.append(retrieved_doc)
        return retrieved_docs
