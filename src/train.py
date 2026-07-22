"""Train every model configuration in config.yaml and log each to MLflow.

    python -m src.train --config configs/config.yaml

Each entry under `experiments:` becomes one MLflow run carrying its
hyperparameters, a description of the data version, all five evaluation
metrics, and the fitted pipeline as an artifact. No hyperparameter is
hardcoded here — everything comes from the YAML file.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import mlflow
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import compute_metrics, confusion, format_metrics, load_config
from src.preprocess import build_preprocessor, load_and_prepare


def build_estimator(model_type: str, params: dict):
    """Instantiate a classifier by name. Unknown names fail loudly."""
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    if model_type == "xgboost":
        # Imported lazily so the rest of the project runs without xgboost.
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    raise ValueError(f"Unknown model_type {model_type!r}")


def build_pipeline(config: dict, model_type: str, params: dict) -> Pipeline:
    """Preprocessing + estimator as one object.

    Bundling them matters twice over: the preprocessor only ever sees training
    data during `fit` (no leakage), and the exported artifact accepts raw
    user-shaped input at serve time, so the Streamlit app cannot drift out of
    sync with training-time encoding.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("classifier", build_estimator(model_type, params)),
        ]
    )


def log_model_artifact(pipeline: Pipeline) -> None:
    """Persist the fitted pipeline as `model/model.joblib` inside the run."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(pipeline, path)
        mlflow.log_artifact(str(path), artifact_path="model")


def run_experiment(experiment: dict, config: dict, data) -> dict[str, float]:
    """Train one configuration and log the whole thing as an MLflow run."""
    X_train, X_test, y_train, y_test = data
    name = experiment["name"]
    model_type = experiment["model_type"]
    params = experiment.get("params") or {}

    with mlflow.start_run(run_name=name):
        pipeline = build_pipeline(config, model_type, params)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline.named_steps["classifier"], "predict_proba")
            else None
        )
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_param("model_type", model_type)
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Data version / description, so a run is reproducible from its log alone.
        mlflow.log_params(
            {
                "data_path": config["data"]["path"],
                "data_rows_train": len(X_train),
                "data_rows_test": len(X_test),
                "data_features": X_train.shape[1],
                "test_size": config["data"]["test_size"],
                "split_random_state": config["data"]["random_state"],
                "scaler": config["preprocessing"]["scaler"],
                "dropped_columns": ",".join(config["preprocessing"].get("drop_columns") or []),
            }
        )
        mlflow.set_tag(
            "data_description",
            "UCI Adult Income (48,842 rows); '?' treated as missing, duplicates "
            "dropped, target binarised to income>50K",
        )

        mlflow.log_metrics(metrics)
        mlflow.log_metrics({k: float(v) for k, v in confusion(y_test, y_pred).items()})

        log_model_artifact(pipeline)

        print(f"  {name:<32} {format_metrics(metrics)}")
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--only",
        help="Run a single named experiment from the config instead of all of them",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    cleaned, data = load_and_prepare(config)
    X_train, X_test = data[0], data[1]
    print(
        f"Loaded {len(cleaned):,} rows -> {len(X_train):,} train / {len(X_test):,} test, "
        f"{X_train.shape[1]} raw features"
    )
    print(f"Positive class rate: {cleaned[config['data']['target']].mean():.1%}\n")

    experiments = config["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            raise SystemExit(f"No experiment named {args.only!r} in {args.config}")

    for experiment in experiments:
        run_experiment(experiment, config, data)

    print(
        f"\nLogged {len(experiments)} run(s) to {config['mlflow']['experiment_name']}. "
        f"Next: python -m src.evaluate"
    )


if __name__ == "__main__":
    main()
