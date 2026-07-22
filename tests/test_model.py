"""Model validation: prediction shape/type, and a performance floor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_metrics
from src.preprocess import feature_columns, load_and_prepare


@pytest.fixture(scope="module")
def held_out(config):
    """The real held-out test split, rebuilt from the same seed as training."""
    try:
        _, (_, X_test, _, y_test) = load_and_prepare(config)
    except FileNotFoundError:
        pytest.skip("dataset not present in data/")
    return X_test, y_test


def test_predictions_have_correct_shape_and_type(trained_model, config):
    """Predictions are binary ints; probabilities are valid and paired."""
    sample = pd.DataFrame(
        [
            {
                "age": 39,
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
            },
            {
                "age": 52,
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 60,
                "workclass": "Self-emp-inc",
                "education": "Masters",
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
            },
        ]
    )[feature_columns(config)]

    predictions = trained_model.predict(sample)
    probabilities = trained_model.predict_proba(sample)

    assert predictions.shape == (2,)
    assert set(np.unique(predictions)) <= {0, 1}
    assert probabilities.shape == (2, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_meets_minimum_performance(trained_model, config, held_out):
    """The exported model clears the accuracy floor set in config.yaml.

    A guard against silently shipping a regressed or mis-exported model.
    """
    X_test, y_test = held_out
    y_pred = trained_model.predict(X_test)
    y_proba = trained_model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    floor = config["evaluation"]["min_accuracy"]

    assert metrics["accuracy"] >= floor, (
        f"accuracy {metrics['accuracy']:.4f} below the {floor} floor"
    )
    # Beating the majority-class baseline (~76% all-negative) requires real
    # signal on the positive class, so check F1 and AUC too.
    assert metrics["f1"] > 0.60
    assert metrics["roc_auc"] > 0.85


def test_higher_earning_profile_scores_higher(trained_model, config):
    """Sanity check that the model learned the expected direction.

    An executive with a master's degree working 60h should not score lower
    than a young part-time worker without a diploma.
    """
    base = {
        "capital-gain": 0,
        "capital-loss": 0,
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
    }
    low = {
        **base,
        "age": 22,
        "hours-per-week": 20,
        "workclass": "Private",
        "education": "11th",
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
    }
    high = {
        **base,
        "age": 50,
        "hours-per-week": 60,
        "workclass": "Self-emp-inc",
        "education": "Masters",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
    }

    frame = pd.DataFrame([low, high])[feature_columns(config)]
    probabilities = trained_model.predict_proba(frame)[:, 1]

    assert probabilities[1] > probabilities[0]
