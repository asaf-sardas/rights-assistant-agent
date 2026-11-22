"""Interfaces ("ports") the rest of the app depends on.

The graph depends on these abstractions, never on concrete implementations.
Any class with a matching `search` method satisfies `Retriever` - no
inheritance needed (structural typing via Protocol).
"""

from typing import Protocol

from rights_assistant_agent.domain.models import RetrievedDoc


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        """Return up to `k` chunks most relevant to `query`, best first."""
        ...
