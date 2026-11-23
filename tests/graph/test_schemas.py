"""Schema tests: valid LLM output parses, invalid output is rejected, and the
JSON schema the LLM sees carries the information it needs."""

from typing import get_args

import pytest
from pydantic import ValidationError

from rights_assistant_agent.domain import Intent, MissingVar
from rights_assistant_agent.graph.schemas import (
    MessageUnderstanding,
    MissingInfoItem,
    ProfileFact,
    ProfileUpdates,
    SufficiencyAssessment,
)

# --- ProfileUpdates -----------------------------------------------------------

TYPED_FIELDS = {
    "status",
    "tenure_months",
    "job_scope",
    "wage_type",
    "sector",
    "age",
    "salary",
}


def test_profile_updates_declares_expected_fields():
    assert set(ProfileUpdates.model_fields) == TYPED_FIELDS | {"other_facts"}


def test_profile_updates_defaults_mean_not_mentioned():
    empty = ProfileUpdates()
    assert all(getattr(empty, name) is None for name in TYPED_FIELDS)
    assert empty.other_facts == []


def test_profile_updates_rejects_unknown_choice():
    with pytest.raises(ValidationError):
        ProfileUpdates(status="on_vacation")


def test_profile_fact_name_must_be_snake_case():
    with pytest.raises(ValidationError):
        ProfileFact(name="Reserve Duty", value="30")


# --- ProfileUpdates.to_profile_dict -------------------------------------------


def test_to_profile_dict_keeps_only_mentioned_typed_fields():
    updates = ProfileUpdates(status="fired", tenure_months=48)
    assert updates.to_profile_dict() == {"status": "fired", "tenure_months": 48}


def test_to_profile_dict_flattens_other_facts():
    updates = ProfileUpdates(
        status="fired",
        other_facts=[ProfileFact(name="reserve_duty_days", value="30")],
    )
    assert updates.to_profile_dict() == {"status": "fired", "reserve_duty_days": "30"}


def test_to_profile_dict_typed_field_wins_over_generic_fact():
    updates = ProfileUpdates(
        tenure_months=48,
        other_facts=[ProfileFact(name="tenure_months", value="about 4 years")],
    )
    assert updates.to_profile_dict() == {"tenure_months": 48}


def test_to_profile_dict_empty_when_nothing_mentioned():
    assert ProfileUpdates().to_profile_dict() == {}


# --- MessageUnderstanding -----------------------------------------------------

LLM_OUTPUT = {
    "intent": "new_topic",
    "topic": "פיצויי פיטורים",
    "search_query": "זכאות לפיצויי פיטורים לאחר 4 שנות עבודה",
    "profile_updates": {"status": "fired", "tenure_months": 48},
}


def test_understanding_parses_llm_output():
    result = MessageUnderstanding.model_validate(LLM_OUTPUT)
    assert result.intent == "new_topic"
    assert result.profile_updates.tenure_months == 48
    assert result.profile_updates.job_scope is None  # not mentioned


def test_understanding_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        MessageUnderstanding.model_validate({**LLM_OUTPUT, "intent": "chit_chat"})


def test_intent_enum_matches_the_domain():
    """The schema must reuse the domain Intent, not a hand-copied list."""
    schema = MessageUnderstanding.model_json_schema()
    assert set(schema["properties"]["intent"]["enum"]) == set(get_args(Intent))


# --- The LLM sees a description for every field -------------------------------


@pytest.mark.parametrize(
    "model",
    [
        ProfileFact,
        ProfileUpdates,
        MessageUnderstanding,
        MissingInfoItem,
        SufficiencyAssessment,
    ],
)
def test_every_field_has_a_description(model):
    missing = [
        name for name, field in model.model_fields.items() if not field.description
    ]
    assert missing == [], f"{model.__name__} fields without description: {missing}"


@pytest.mark.parametrize(
    "model",
    [
        ProfileFact,
        ProfileUpdates,
        MessageUnderstanding,
        MissingInfoItem,
        SufficiencyAssessment,
    ],
)
def test_docstring_is_an_instruction_not_a_todo(model):
    assert model.__doc__ and "TODO" not in model.__doc__


# --- SufficiencyAssessment ----------------------------------------------------


def test_importance_must_be_positive():
    with pytest.raises(ValidationError):
        MissingInfoItem(name="age", question="בן כמה אתה?", importance=0)


def test_to_missing_vars_returns_domain_objects_sorted_by_importance():
    assessment = SufficiencyAssessment(
        sources_cover_question=True,
        missing_info=[
            MissingInfoItem(name="job_scope", question="משרה מלאה?", importance=2),
            MissingInfoItem(
                name="tenure_months", question="כמה זמן עבדת?", importance=1
            ),
        ],
    )
    assert assessment.to_missing_vars() == [
        MissingVar("tenure_months", "כמה זמן עבדת?", 1),
        MissingVar("job_scope", "משרה מלאה?", 2),
    ]


def test_to_missing_vars_empty_when_nothing_missing():
    assessment = SufficiencyAssessment(sources_cover_question=True, missing_info=[])
    assert assessment.to_missing_vars() == []
