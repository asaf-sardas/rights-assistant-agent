"""The graph state: everything the agent remembers between turns.

Nodes receive the whole state and return only the keys they change.
LangGraph merges each update into the state using the field's reducer:
- no reducer           -> the new value overwrites the old one
- Annotated[T, reducer] -> reducer(old_value, update) decides the new value

The checkpointer saves the state after every run, keyed by thread_id.

Fields a node hasn't written yet are simply missing from the state dict,
so nodes should read optional fields with state.get("field", default).
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from rights_assistant_agent.domain import Intent, MissingVar, Phase, RetrievedDoc


def merge_profile(
    old: dict[str, Any] | None, new: dict[str, Any] | None
) -> dict[str, Any]:
    """Reducer for `user_profile`: merge newly learned facts into the existing profile.

    Rules:
    - Never lose existing facts: keys only in `old` are kept.
    - A new value for an existing key replaces the old value
      (the user corrected themselves: "actually it's 4 years, not 3").
    - A value of None in `new` means "not mentioned" and must NOT erase the old value.
    - `old` or `new` may be None (e.g. on the very first update) -> treat as {}.
    - Return a NEW dict; don't modify `old` or `new` in place.
    """
    profile = dict(old or {})
    for key, value in (new or {}).items():
        if value is not None:
            profile[key] = value
    return profile


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_profile: Annotated[dict[str, Any], merge_profile]
    intent: Intent | None
    topic: str | None
    search_query: str | None
    retrieved_docs: list[RetrievedDoc]
    sources_cover_question: bool
    missing_info: list[MissingVar]
    clarification_rounds: int
    phase: Phase
