"""Shared fixtures. Keeps the unit tests independent of the full dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import load_config  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A tiny hand-built frame mirroring the real schema.

    Row 3 carries missing `workclass`/`occupation` (the dataset's "?"), and
    rows 0 and 4 are exact duplicates, so cleaning and imputation are both
    exercised without touching the 48K-row file.
    """
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 39],
            "workclass": ["State-gov", "Self-emp-not-inc", "Private", None, "State-gov"],
            "fnlwgt": [77516, 83311, 215646, 234721, 77516],
            "education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors"],
            "education-num": [13, 13, 9, 7, 13],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Never-married",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                None,
                "Adm-clerical",
            ],
            "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband", "Not-in-family"],
            "race": ["White", "White", "White", "Black", "White"],
            "sex": ["Male", "Male", "Male", "Male", "Male"],
            "capital-gain": [2174, 0, 0, 0, 2174],
            "capital-loss": [0, 0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40, 40],
            "native-country": ["United-States"] * 5,
            "income": ["<=50K", "<=50K", "<=50K", ">50K", "<=50K"],
        }
    )


@pytest.fixture(scope="session")
def trained_model():
    """The exported best model. Skips if training hasn't been run yet."""
    if not MODEL_PATH.exists():
        pytest.skip("models/best_model.joblib missing — run train + evaluate first")
    import joblib

    return joblib.load(MODEL_PATH)
