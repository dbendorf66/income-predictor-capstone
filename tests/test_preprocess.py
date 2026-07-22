"""Preprocessing unit tests: missing values, encoding, scaling, immutability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    build_preprocessor,
    clean_data,
    feature_columns,
    normalize_columns,
    split_data,
)


def test_clean_data_does_not_mutate_input(raw_frame, config):
    """clean_data must return a new frame and leave the caller's untouched."""
    before = raw_frame.copy(deep=True)

    cleaned = clean_data(
        raw_frame,
        target=config["data"]["target"],
        positive_label=config["data"]["positive_label"],
        drop_columns=config["preprocessing"]["drop_columns"],
    )

    pd.testing.assert_frame_equal(raw_frame, before)
    assert cleaned is not raw_frame
    # The target really was transformed on the copy.
    assert set(cleaned["income"].unique()) <= {0, 1}


def test_clean_data_handles_missing_and_duplicates(raw_frame, config):
    """Missing features survive as NaN (for the imputer); duplicates go."""
    cleaned = clean_data(
        raw_frame,
        target="income",
        positive_label=">50K",
        drop_columns=config["preprocessing"]["drop_columns"],
    )

    # Rows 0 and 4 were identical -> 5 rows in, 4 out.
    assert len(cleaned) == 4
    # Dropped columns are gone.
    assert "fnlwgt" not in cleaned.columns
    assert "education-num" not in cleaned.columns
    # Feature-level missingness is preserved for the pipeline's imputer,
    # NOT dropped here — that keeps train and serve paths identical.
    assert cleaned["occupation"].isna().sum() == 1


def test_preprocessor_imputes_encodes_and_scales(raw_frame, config):
    """Fitted transform: no NaNs, one-hot widening, standardised numerics."""
    cleaned = clean_data(
        raw_frame,
        target="income",
        positive_label=">50K",
        drop_columns=config["preprocessing"]["drop_columns"],
    )
    X = cleaned[feature_columns(config)]

    preprocessor = build_preprocessor(config)
    transformed = preprocessor.fit_transform(X)

    # Imputation filled every hole.
    assert not np.isnan(transformed).any()
    # One-hot encoding widened the frame beyond the 12 raw columns.
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > X.shape[1]

    # StandardScaler puts the numeric block at ~zero mean, unit-ish variance.
    n_numeric = len(config["preprocessing"]["numeric_features"])
    numeric_block = transformed[:, :n_numeric]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-7)

    # One-hot block is strictly 0/1.
    categorical_block = transformed[:, n_numeric:]
    assert np.isin(categorical_block, [0.0, 1.0]).all()


def test_preprocessor_ignores_unseen_categories(raw_frame, config):
    """An unseen category at serve time must degrade, not raise.

    This is the exact path a Streamlit user hits when the LLM supplies a
    category the training split never contained.
    """
    cleaned = clean_data(
        raw_frame, target="income", positive_label=">50K",
        drop_columns=config["preprocessing"]["drop_columns"],
    )
    X = cleaned[feature_columns(config)]

    preprocessor = build_preprocessor(config).fit(X)

    unseen = X.iloc[[0]].copy()
    unseen.loc[unseen.index[0], "native-country"] = "Atlantis"
    transformed = preprocessor.transform(unseen)

    assert transformed.shape[1] == preprocessor.transform(X.iloc[[0]]).shape[1]
    assert not np.isnan(transformed).any()


def test_normalize_columns_unifies_spellings():
    """Kaggle re-uploads use other spellings; all map to one canonical form."""
    assert normalize_columns(
        ["Age", "marital.status", "capital_gain", "educational-num", "class"]
    ) == ["age", "marital-status", "capital-gain", "education-num", "income"]


def test_split_is_stratified_and_disjoint(config):
    """Train/test split preserves class balance and shares no rows."""
    frame = pd.DataFrame(
        {
            "age": list(range(100)),
            "capital-gain": [0] * 100,
            "capital-loss": [0] * 100,
            "hours-per-week": [40] * 100,
            "workclass": ["Private"] * 100,
            "education": ["HS-grad"] * 100,
            "marital-status": ["Divorced"] * 100,
            "occupation": ["Sales"] * 100,
            "relationship": ["Husband"] * 100,
            "race": ["White"] * 100,
            "sex": ["Male"] * 100,
            "native-country": ["United-States"] * 100,
            "income": [1] * 25 + [0] * 75,
        }
    )

    X_train, X_test, y_train, y_test = split_data(frame, config)

    assert len(X_train) + len(X_test) == 100
    assert set(X_train.index).isdisjoint(X_test.index)
    # 25% positive in both folds, because the split is stratified.
    assert y_train.mean() == pytest.approx(0.25, abs=0.02)
    assert y_test.mean() == pytest.approx(0.25, abs=0.02)
