"""Data loading and preprocessing for the Adult Income dataset.

The public entry points are:

    load_raw_data(path, ...)      -> tidy DataFrame straight off disk
    clean_data(df, ...)           -> missing values handled, target normalised
    build_preprocessor(config)    -> unfitted sklearn ColumnTransformer
    split_data(df, config)        -> stratified train/test split

Two design decisions are worth calling out:

1. All fitted transformations (imputation, scaling, one-hot vocabularies) live
   inside a sklearn ColumnTransformer that is fitted **only on the training
   fold**, then bundled into the same Pipeline as the estimator. That is what
   prevents test-set statistics from leaking into training.
2. Every function here treats its input DataFrame as read-only and returns a
   new object. Callers can chain them without surprise mutations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Canonical column order of the original UCI `adult.data` file, which ships
# without a header row.
UCI_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


def normalize_columns(columns) -> list[str]:
    """Map assorted spellings of the Adult columns onto one canonical form.

    Kaggle re-uploads of this dataset use `marital.status`, `educational-num`,
    `capital_gain` and friends. Normalising to lowercase hyphenated names means
    the rest of the codebase (and config.yaml) only has to know one spelling.
    """
    renamed = []
    for col in columns:
        clean = str(col).strip().lower().replace(".", "-").replace("_", "-")
        if clean == "educational-num":
            clean = "education-num"
        if clean in ("class", "salary", "target"):
            clean = "income"
        renamed.append(clean)
    return renamed


def _read_one(path: Path, na_value: str, columns: list[str]) -> pd.DataFrame:
    """Read a single Adult CSV, with or without a header row.

    The original UCI files are headerless and use ", " as their separator;
    Kaggle re-exports are ordinary CSVs with a header. This sniffs which one it
    got and handles both. `comment="|"` drops the documentation banner that
    `adult.test` carries on its first line.
    """
    first_line = ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() and not line.startswith("|"):
                first_line = line
                break
    has_header = "age" in first_line.lower()

    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else columns,
        sep=",",
        skipinitialspace=True,
        skip_blank_lines=True,
        comment="|",
        na_values=[na_value, "", " "],
        engine="python",
    )
    df.columns = normalize_columns(df.columns)
    return df


def load_raw_data(
    path: str | Path,
    na_value: str = "?",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the Adult dataset from a file, or from a directory of UCI files.

    Pointing at a directory concatenates `adult.data` and `adult.test` into the
    full 48,842-row dataset. We re-split it ourselves rather than using the
    canonical UCI split so that the held-out set is stratified and the split
    seed is recorded in config.yaml.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Adult Income dataset "
            f"and place it there (see README, 'Getting the data')."
        )

    columns = columns or UCI_COLUMNS

    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix in (".data", ".test", ".csv")
        )
        if not files:
            raise FileNotFoundError(
                f"No .data/.test/.csv files found in {path}. Expected the UCI "
                f"`adult.data` and `adult.test` files."
            )
        df = pd.concat(
            [_read_one(f, na_value, columns) for f in files], ignore_index=True
        )
    else:
        df = _read_one(path, na_value, columns)

    # `adult.test` writes its label as `<=50K.` with a trailing period.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.rstrip(".")

    return df


def clean_data(
    df: pd.DataFrame,
    target: str = "income",
    positive_label: str = ">50K",
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop unusable rows/columns and binarise the target.

    Rows with a missing *target* are dropped — they cannot be trained or
    scored on. Missing *features* are deliberately left as NaN here so the
    imputers inside the fitted pipeline handle them, keeping train and serve
    paths identical.

    Returns a new DataFrame; `df` is not modified.
    """
    out = df.copy()

    if target not in out.columns:
        raise KeyError(f"Target column {target!r} not found in {list(out.columns)}")

    out = out.dropna(subset=[target])

    # Deduplicate on the *full* record, before any columns are dropped. Doing
    # it afterwards would collapse genuinely distinct census respondents who
    # happen to share demographics once `fnlwgt` is gone — that discards ~13%
    # of the data and is not what "duplicate" should mean here.
    out = out.drop_duplicates()

    for col in drop_columns or []:
        if col in out.columns:
            out = out.drop(columns=col)

    # `>50K` -> 1, `<=50K` -> 0.
    out[target] = (
        out[target].astype(str).str.strip().str.rstrip(".").eq(positive_label).astype(int)
    )

    return out.reset_index(drop=True)


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Assemble the unfitted feature pipeline described by `config`.

    Numeric branch:     impute -> scale
    Categorical branch: impute -> one-hot (unknown categories ignored at serve
                        time, so an unseen `native-country` degrades to all-zeros
                        instead of raising).
    """
    pre_cfg = config["preprocessing"]

    scaler_name = str(pre_cfg.get("scaler", "standard")).lower()
    if scaler_name not in _SCALERS:
        raise ValueError(
            f"Unknown scaler {scaler_name!r}; expected one of {sorted(_SCALERS)}"
        )
    scaler_cls = _SCALERS[scaler_name]

    numeric_steps = [
        ("impute", SimpleImputer(strategy=pre_cfg.get("numeric_imputer", "median")))
    ]
    if scaler_cls is not None:
        numeric_steps.append(("scale", scaler_cls()))

    categorical_steps = [
        (
            "impute",
            SimpleImputer(strategy=pre_cfg.get("categorical_imputer", "most_frequent")),
        ),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), pre_cfg["numeric_features"]),
            ("categorical", Pipeline(categorical_steps), pre_cfg["categorical_features"]),
        ],
        remainder="drop",
    )


def feature_columns(config: dict) -> list[str]:
    """The exact feature names, in order, that a model expects at inference."""
    pre_cfg = config["preprocessing"]
    return list(pre_cfg["numeric_features"]) + list(pre_cfg["categorical_features"])


def split_data(df: pd.DataFrame, config: dict):
    """Stratified train/test split. Returns (X_train, X_test, y_train, y_test)."""
    data_cfg = config["data"]
    target = data_cfg["target"]

    X = df[feature_columns(config)]
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
        stratify=y,
    )


def load_and_prepare(config: dict):
    """Convenience wrapper: disk -> cleaned frame -> stratified split."""
    data_cfg = config["data"]
    raw = load_raw_data(data_cfg["path"], na_value=data_cfg.get("na_value", "?"))
    cleaned = clean_data(
        raw,
        target=data_cfg["target"],
        positive_label=data_cfg.get("positive_label", ">50K"),
        drop_columns=config["preprocessing"].get("drop_columns"),
    )
    return cleaned, split_data(cleaned, config)
