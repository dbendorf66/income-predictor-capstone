"""Natural-language layer over the trained income classifier.

The LLM is the *interface*, not the predictor. The flow is a two-call tool-use
loop against the Anthropic Messages API:

    1. The user's plain-English question goes to Claude with a `predict_income`
       tool whose schema mirrors the model's feature space exactly.
    2. Claude either calls the tool with extracted feature values, or replies
       with text (asking for missing details, or declining an out-of-scope
       question). Deciding *that* is the whole point of using an LLM here.
    3. If it called the tool, we run the real trained pipeline, hand the
       prediction back as a tool result, and Claude explains it in context.

The predictions themselves always come from `models/best_model.joblib` — the
LLM never guesses a number.

This module holds all the logic and no UI, so `tests/test_interface.py` can
exercise parsing and edge cases without Streamlit or a live API key.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/best_model.joblib")

# Without at least these, a prediction would be driven almost entirely by
# imputed defaults, which is exactly the "garbage prediction" the spec warns
# against. Everything else falls back to a training-set default and is
# reported to the user as an assumption.
REQUIRED_FEATURES = (
    "age",
    "education",
    "marital-status",
    "occupation",
    "hours-per-week",
)

SYSTEM_PROMPT = """\
You are the interface to a machine learning model that predicts whether a US \
adult's income exceeds $50,000/year. The model was trained on the 1994 UCI \
Adult Census dataset.

Your job:
1. Read the user's plain-English description of a person.
2. If you have enough information, call the `predict_income` tool with the \
feature values you extracted. Map the user's wording onto the allowed \
category values — for example "I'm a software engineer" is "Prof-specialty", \
"I finished my bachelor's" is "Bachelors", "I work 50 hours a week" is 50.
3. When the tool returns a prediction, explain it in plain language: state the \
predicted class and probability, say which details mattered most, and flag \
any values you had to assume.

Rules:
- NEVER invent a prediction. The number must come from the tool.
- If required details are missing (age, education, marital status, \
occupation, or weekly hours), do NOT call the tool. Ask for exactly what you \
need, in one short question.
- If the question is not about predicting a person's income, say so briefly \
and explain what you can help with instead. Do not call the tool.
- Always mention the caveats: the model reflects 1994 US census data, so its \
dollar threshold and its social patterns are dated, and it reproduces the \
biases present in that data. It is a demonstration, not financial advice.
"""


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Load the exported best pipeline (preprocessing + classifier)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train` then "
            f"`python -m src.evaluate` first."
        )
    return joblib.load(path)


def feature_space(model) -> dict:
    """Derive the model's accepted features from the *fitted* pipeline.

    Reading categories and defaults off the fitted transformers — rather than
    hardcoding them here — means the tool schema can never drift out of sync
    with what the model was actually trained on. The OneHotEncoder knows its
    vocabulary and the SimpleImputers know their learned fallback values.
    """
    preprocessor = model.named_steps["preprocessor"]
    space: dict[str, dict] = {"numeric": {}, "categorical": {}}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            medians = transformer.named_steps["impute"].statistics_
            for column, median in zip(columns, medians):
                space["numeric"][column] = float(median)
        elif name == "categorical":
            modes = transformer.named_steps["impute"].statistics_
            categories = transformer.named_steps["encode"].categories_
            for column, mode, values in zip(columns, modes, categories):
                space["categorical"][column] = {
                    "categories": [str(v) for v in values],
                    "default": str(mode),
                }

    return space


def build_tool(space: dict) -> dict:
    """Build the `predict_income` tool schema from the feature space."""
    properties: dict[str, dict] = {}

    for column in space["numeric"]:
        properties[column] = {
            "type": "number",
            "description": f"The person's {column.replace('-', ' ')}.",
        }
    for column, meta in space["categorical"].items():
        properties[column] = {
            "type": "string",
            "enum": meta["categories"],
            "description": (
                f"The person's {column.replace('-', ' ')}. Must be one of the "
                f"listed values — map the user's wording onto the closest one."
            ),
        }

    return {
        "name": "predict_income",
        "description": (
            "Run the trained classifier on one person's demographic and "
            "employment details. Returns the predicted income class and the "
            "model's probability. Only call this once you have the required "
            "fields; omit optional fields you genuinely do not know rather "
            "than guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(REQUIRED_FEATURES),
        },
    }


def resolve_features(tool_input: dict, space: dict) -> tuple[dict, list[str]]:
    """Fill omitted optional features with training defaults.

    Returns `(features, assumptions)` where `assumptions` names every field
    that was defaulted, so the response can be transparent about it. Unknown
    keys are dropped and out-of-vocabulary categories fall back to the default
    rather than reaching the encoder as an unseen value.
    """
    features: dict = {}
    assumptions: list[str] = []

    for column, default in space["numeric"].items():
        value = tool_input.get(column)
        if value is None:
            features[column] = default
            assumptions.append(f"{column} = {default:g} (dataset median)")
        else:
            features[column] = float(value)

    for column, meta in space["categorical"].items():
        value = tool_input.get(column)
        if value is None or str(value) not in meta["categories"]:
            features[column] = meta["default"]
            assumptions.append(f"{column} = {meta['default']} (most common)")
        else:
            features[column] = str(value)

    return features, assumptions


def missing_required(tool_input: dict) -> list[str]:
    """Required fields absent from a tool call — used to guard bad calls."""
    return [f for f in REQUIRED_FEATURES if tool_input.get(f) is None]


def predict(model, features: dict) -> dict:
    """Run the trained pipeline on one resolved feature dict."""
    frame = pd.DataFrame([features])
    label = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "prediction": ">50K" if label else "<=50K",
        "probability_over_50k": round(probability, 4),
        "label": label,
    }


def build_client(api_key: str | None = None):
    """Construct the Anthropic client. Key comes from the environment."""
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key (never hardcode it in source)."
        )
    return anthropic.Anthropic(api_key=key)


def answer(client, model, query: str, history: list | None = None) -> dict:
    """Run one full turn: parse -> predict -> explain.

    Returns a dict with the assistant's `text`, the `prediction` (None when the
    model was never invoked), the `features` used, any `assumptions` made, and
    the updated `messages` list for multi-turn use.
    """
    llm_model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
    space = feature_space(model)
    tool = build_tool(space)

    messages = list(history or []) + [{"role": "user", "content": query}]

    response = client.messages.create(
        model=llm_model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[tool],
        messages=messages,
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)

    # No tool call: Claude asked a clarifying question or declined an
    # out-of-scope request. Pass that straight through.
    if tool_use is None:
        text = "".join(b.text for b in response.content if b.type == "text")
        messages.append({"role": "assistant", "content": response.content})
        return {
            "text": text,
            "prediction": None,
            "features": None,
            "assumptions": [],
            "messages": messages,
        }

    missing = missing_required(tool_use.input)
    if missing:
        # Defensive: the system prompt forbids this, but a model can still
        # call early. Report the gap instead of predicting on defaults.
        result_payload = {
            "error": "missing_required_features",
            "missing": missing,
            "message": "Ask the user for these fields before predicting.",
        }
        features, assumptions, prediction = None, [], None
    else:
        features, assumptions = resolve_features(tool_use.input, space)
        prediction = predict(model, features)
        result_payload = {**prediction, "assumed_defaults": assumptions}

    messages.append({"role": "assistant", "content": response.content})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(result_payload),
                    "is_error": bool(missing),
                }
            ],
        }
    )

    follow_up = client.messages.create(
        model=llm_model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[tool],
        messages=messages,
    )
    text = "".join(b.text for b in follow_up.content if b.type == "text")
    messages.append({"role": "assistant", "content": follow_up.content})

    return {
        "text": text,
        "prediction": prediction,
        "features": features,
        "assumptions": assumptions,
        "messages": messages,
    }
