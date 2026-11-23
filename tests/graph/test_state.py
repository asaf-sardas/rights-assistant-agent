"""State tests: the profile reducer, and how LangGraph merges node updates."""

from typing import get_type_hints

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from rights_assistant_agent.graph.state import AgentState, merge_profile

# --- merge_profile (a plain function: easy to test directly) -----------------


def test_merge_adds_new_facts_and_keeps_old_ones():
    assert merge_profile({"tenure_years": 3}, {"wage_type": "monthly"}) == {
        "tenure_years": 3,
        "wage_type": "monthly",
    }


def test_merge_new_value_replaces_old_value():
    assert merge_profile({"tenure_years": 3}, {"tenure_years": 4}) == {
        "tenure_years": 4
    }


def test_merge_none_does_not_erase_a_known_fact():
    assert merge_profile({"tenure_years": 3}, {"tenure_years": None}) == {
        "tenure_years": 3
    }


def test_merge_handles_missing_sides():
    assert merge_profile(None, {"a": 1}) == {"a": 1}
    assert merge_profile({"a": 1}, None) == {"a": 1}
    assert merge_profile(None, None) == {}


def test_merge_does_not_modify_its_inputs():
    old, new = {"a": 1}, {"b": 2}
    merge_profile(old, new)
    assert old == {"a": 1}
    assert new == {"b": 2}


# --- AgentState declares all fields -------------------------------------------

EXPECTED_FIELDS = {
    "messages",
    "user_profile",
    "intent",
    "topic",
    "search_query",
    "retrieved_docs",
    "sources_cover_question",
    "missing_info",
    "clarification_rounds",
    "phase",
}


def test_state_declares_all_fields():
    assert set(get_type_hints(AgentState)) == EXPECTED_FIELDS


# --- The reducers in action inside a real (tiny) graph ------------------------


def run_one_node_graph(initial: dict, node_update: dict) -> dict:
    """Build a graph with a single node that returns `node_update`, run it once."""
    builder = StateGraph(AgentState)
    builder.add_node("node", lambda state: node_update)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile().invoke(initial)


def test_messages_are_appended_not_replaced():
    result = run_one_node_graph(
        initial={"messages": [HumanMessage("שלום")]},
        node_update={"messages": [AIMessage("היי, איך אפשר לעזור?")]},
    )
    assert [m.content for m in result["messages"]] == ["שלום", "היי, איך אפשר לעזור?"]


def test_profile_is_merged_not_replaced():
    result = run_one_node_graph(
        initial={"user_profile": {"tenure_years": 3}},
        node_update={"user_profile": {"wage_type": "monthly"}},
    )
    assert result["user_profile"] == {"tenure_years": 3, "wage_type": "monthly"}


def test_plain_fields_are_overwritten():
    result = run_one_node_graph(
        initial={"phase": "new", "topic": "פיצויים"},
        node_update={"phase": "clarifying"},
    )
    assert result["phase"] == "clarifying"
    assert result["topic"] == "פיצויים"  # untouched keys stay as they were
