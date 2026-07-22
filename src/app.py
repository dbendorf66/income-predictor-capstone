"""Streamlit chat interface for the Adult Income classifier.

    streamlit run src/app.py

All the logic lives in `src/llm_interface.py`; this file is presentation only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow `streamlit run src/app.py` from the project root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_interface import answer, build_client, load_model  # noqa: E402

load_dotenv()

st.set_page_config(page_title="Income Predictor", page_icon="💰")

EXAMPLES = [
    "I'm a 39-year-old married software engineer with a bachelor's degree, "
    "working 45 hours a week. What does the model predict?",
    "I'm 22 and work part time.",
    "What's the weather in Chicago?",
]


@st.cache_resource
def _load():
    """Load the trained pipeline and API client once per server process."""
    return load_model(), build_client()


st.title("💰 Income Predictor")
st.caption(
    "Ask in plain English. A trained XGBoost classifier makes the prediction; "
    "Claude parses your question and explains the result."
)

try:
    model, client = _load()
except Exception as exc:  # noqa: BLE001 — surface setup errors in the UI
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "**Model:** XGBoost on the UCI Adult Census dataset (48,790 rows).\n\n"
        "**Test performance:** 88.0% accuracy, 0.731 F1, 0.933 ROC-AUC.\n\n"
        "The model predicts whether income exceeds **$50,000/year** in "
        "**1994** dollars. It reflects the biases of that census data and is "
        "a demonstration, not financial advice."
    )
    st.caption(f"LLM: {os.environ.get('ANTHROPIC_MODEL', 'claude-opus-4-8')}")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.transcript = []
        st.rerun()

    st.subheader("Try asking")
    for example in EXAMPLES:
        st.caption(f"• {example}")

st.session_state.setdefault("history", [])
st.session_state.setdefault("transcript", [])

for role, text in st.session_state.transcript:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Describe a person, or ask about their income..."):
    st.session_state.transcript.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer(client, model, prompt, st.session_state.history)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
                st.stop()

        st.markdown(result["text"])
        st.session_state.history = result["messages"]
        st.session_state.transcript.append(("assistant", result["text"]))

        if result["prediction"]:
            probability = result["prediction"]["probability_over_50k"]
            st.metric("P(income > $50K)", f"{probability:.1%}")
            st.progress(probability)
            with st.expander("Features the model actually used"):
                st.json(result["features"])
                if result["assumptions"]:
                    st.caption("Assumed defaults: " + "; ".join(result["assumptions"]))
