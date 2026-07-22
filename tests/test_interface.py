"""Interface tests: feature parsing and graceful edge-case handling.

These use a stub Anthropic client so the suite runs offline and costs nothing.
The stub replays the exact content-block shapes the real SDK returns, so the
tool-use loop in `answer()` is genuinely exercised.
"""

from __future__ import annotations

import pytest

from src.llm_interface import (
    answer,
    build_tool,
    feature_space,
    missing_required,
    predict,
    resolve_features,
)


class _Block:
    """Stand-in for an SDK content block (`text` or `tool_use`)."""

    def __init__(self, type_, **kwargs):
        self.type = type_
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Response:
    def __init__(self, content):
        self.content = content


class StubClient:
    """Replays a scripted list of responses and records the requests made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


@pytest.fixture
def space(trained_model):
    return feature_space(trained_model)


def test_tool_schema_matches_model_feature_space(trained_model, space):
    """The tool the LLM sees must mirror what the pipeline actually accepts."""
    tool = build_tool(space)
    properties = tool["input_schema"]["properties"]

    expected = set(space["numeric"]) | set(space["categorical"])
    assert set(properties) == expected

    # Categorical fields are constrained to the encoder's real vocabulary,
    # so the LLM cannot invent a category the model never saw.
    assert "Bachelors" in properties["education"]["enum"]
    assert properties["age"]["type"] == "number"
    assert set(tool["input_schema"]["required"]) <= expected


def test_parses_natural_language_into_correct_feature_values(trained_model, space):
    """A realistic tool call resolves to the exact features the model runs on."""
    tool_input = {
        "age": 39,
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "hours-per-week": 45,
    }

    features, assumptions = resolve_features(tool_input, space)

    # Values the user supplied survive verbatim, with correct types.
    assert features["age"] == 39.0
    assert features["education"] == "Bachelors"
    assert features["occupation"] == "Prof-specialty"
    assert features["hours-per-week"] == 45.0

    # Every model input is populated, and omissions are disclosed.
    assert set(features) == set(space["numeric"]) | set(space["categorical"])
    assert any("native-country" in a for a in assumptions)

    # The resolved dict is directly runnable against the real model.
    result = predict(trained_model, features)
    assert result["prediction"] in {">50K", "<=50K"}
    assert 0.0 <= result["probability_over_50k"] <= 1.0


def test_out_of_vocabulary_category_falls_back_to_default(space):
    """A bad category is replaced and reported, never passed through."""
    features, assumptions = resolve_features(
        {
            "age": 30,
            "education": "Wizardry",  # not a real category
            "marital-status": "Divorced",
            "occupation": "Sales",
            "hours-per-week": 40,
        },
        space,
    )

    assert features["education"] in space["categorical"]["education"]["categories"]
    assert features["education"] == space["categorical"]["education"]["default"]
    assert any("education" in a for a in assumptions)


def test_incomplete_input_asks_instead_of_predicting(trained_model):
    """Missing required fields must produce a question, not a guess."""
    assert missing_required({"age": 22}) == [
        "education",
        "marital-status",
        "occupation",
        "hours-per-week",
    ]

    client = StubClient(
        [
            _Response(
                [
                    _Block(
                        "text",
                        text="I need a bit more detail — what's your education "
                        "level, occupation, marital status, and weekly hours?",
                    )
                ]
            )
        ]
    )

    result = answer(client, trained_model, "I'm 22 and work part time.")

    assert result["prediction"] is None
    assert result["features"] is None
    assert "education" in result["text"]
    # Exactly one API call: it never reached the predict-and-explain round trip.
    assert len(client.requests) == 1


def test_out_of_scope_query_is_declined_without_prediction(trained_model):
    """Off-topic questions short-circuit before the model is invoked."""
    client = StubClient(
        [
            _Response(
                [
                    _Block(
                        "text",
                        text="I can't help with the weather. I predict whether "
                        "a person's income exceeds $50K from census details.",
                    )
                ]
            )
        ]
    )

    result = answer(client, trained_model, "What's the weather in Chicago?")

    assert result["prediction"] is None
    assert "income" in result["text"]


def test_full_tool_use_loop_invokes_the_real_model(trained_model):
    """End-to-end: tool call -> real prediction -> explanation."""
    client = StubClient(
        [
            _Response(
                [
                    _Block(
                        "tool_use",
                        id="toolu_test123",
                        name="predict_income",
                        input={
                            "age": 50,
                            "education": "Masters",
                            "marital-status": "Married-civ-spouse",
                            "occupation": "Exec-managerial",
                            "hours-per-week": 60,
                            "workclass": "Self-emp-inc",
                            "relationship": "Husband",
                        },
                    )
                ]
            ),
            _Response([_Block("text", text="The model predicts over $50K.")]),
        ]
    )

    result = answer(client, trained_model, "I'm a 50-year-old married executive...")

    assert result["prediction"] is not None
    assert result["prediction"]["prediction"] in {">50K", "<=50K"}
    assert result["features"]["age"] == 50.0
    assert "50" in result["text"]

    # Two calls, and the second carried the tool_result back to the model.
    assert len(client.requests) == 2

    # Scan for the tool_result block rather than indexing the tail: `answer`
    # keeps appending to the same list, so the final entry has since become
    # the follow-up assistant turn.
    blocks = [
        block
        for message in client.requests[1]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(blocks) == 1
    assert blocks[0]["tool_use_id"] == "toolu_test123"
    assert not blocks[0]["is_error"]
