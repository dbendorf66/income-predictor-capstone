"""Evaluation metrics and MLflow run comparison.

Run directly to rank every logged experiment and export the winner:

    python -m src.evaluate --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Five classification metrics; `roc_auc` is omitted when no probabilities.

    `zero_division=0` keeps a degenerate model (predicting one class only) from
    raising instead of scoring badly, which is what we want during a sweep.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def confusion(y_true, y_pred) -> dict[str, int]:
    """Flat confusion-matrix counts, convenient for MLflow metric logging."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def search_best_run(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Query MLflow for all runs in the experiment and rank them.

    Returns `(leaderboard, best_run)`. The leaderboard is a tidy DataFrame with
    one row per run, sorted best-first on the configured primary metric.
    """
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    primary = config["evaluation"]["primary_metric"]

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment {experiment_name!r} does not exist yet. "
            f"Run `python -m src.train` first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{primary} DESC"],
    )
    if runs.empty:
        raise RuntimeError(
            f"No finished runs found in {experiment_name!r}. Run `python -m src.train`."
        )

    columns = ["run_id", "tags.mlflow.runName", "params.model_type"] + [
        f"metrics.{m}" for m in METRIC_NAMES if f"metrics.{m}" in runs.columns
    ]
    leaderboard = runs[[c for c in columns if c in runs.columns]].rename(
        columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.", "").replace("params.", "")
    )

    return leaderboard, runs.iloc[0]


def export_best_model(config: dict, best_run: pd.Series) -> Path:
    """Copy the winning run's model artifact to `models/best_model.joblib`.

    The Streamlit app loads this single stable path, so it never has to know
    anything about MLflow run IDs.
    """
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / config["output"]["best_model_name"]

    local_path = mlflow.artifacts.download_artifacts(
        run_id=best_run["run_id"], artifact_path="model/model.joblib"
    )
    shutil.copyfile(local_path, destination)

    metadata = {
        "run_id": best_run["run_id"],
        "run_name": best_run.get("tags.mlflow.runName"),
        "model_type": best_run.get("params.model_type"),
        "primary_metric": config["evaluation"]["primary_metric"],
        "metrics": {
            m: float(best_run[f"metrics.{m}"])
            for m in METRIC_NAMES
            if f"metrics.{m}" in best_run and not pd.isna(best_run[f"metrics.{m}"])
        },
    }
    with open(model_dir / "best_model_metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Print the leaderboard without writing models/best_model.joblib",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    leaderboard, best_run = search_best_run(config)

    primary = config["evaluation"]["primary_metric"]
    print(f"\nExperiment leaderboard (ranked by {primary}):\n")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(leaderboard.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    name = best_run.get("tags.mlflow.runName", best_run["run_id"])
    print(f"\nBest run: {name}  ({primary}={best_run[f'metrics.{primary}']:.4f})")

    if not args.no_export:
        destination = export_best_model(config, best_run)
        print(f"Exported best model -> {destination}")


if __name__ == "__main__":
    main()
