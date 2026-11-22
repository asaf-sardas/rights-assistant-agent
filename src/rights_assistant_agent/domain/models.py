"""Core data shapes shared by all layers.

Pure Python (stdlib only) so that every other layer can depend on it
without pulling in LangChain, OpenAI or Chroma.
"""

from dataclasses import dataclass
from typing import Literal

# What the user's latest message is doing in the conversation.
Intent = Literal["new_topic", "clarification_answer", "follow_up", "out_of_scope"]

# Where the conversation stands on the current topic.
Phase = Literal["new", "clarifying", "answered"]


@dataclass(frozen=True)
class RetrievedDoc:
    """One chunk of a Kol Zchut page returned by a search.

    Its citation id is its position in the results list (1-based),
    so the retriever doesn't need to know about citations.
    """

    title: str
    url: str
    text: str
    score: float  # similarity; higher = more relevant


@dataclass(frozen=True)
class MissingVar:
    """A detail about the user we still need in order to answer precisely."""

    name: str  # e.g. "tenure_months"
    question: str  # the question to ask the user (Hebrew)
    importance: int  # 1 = most important; used to order the questions
