# Income Predictor — End-to-End ML + LLM Application

A natural-language interface over a trained income classifier. You describe a
person in plain English; an LLM extracts the structured features, a trained
XGBoost model makes the prediction, and the LLM explains the result in context.

**What problem it solves.** A gradient-boosted model over 12 census features is
accurate but unusable by a non-specialist — it expects an exact feature vector
with exact category spellings (`Prof-specialty`, not "software engineer"). This
project puts a language layer in front of it, so a user can ask a question the
way they'd ask a person while the prediction still comes from the real model.

**Who it's for.** Anyone who wants a model's answer without learning its schema —
and, as a reference, anyone building an LLM front-end over a tabular model.

---

## Architecture

```
User: "I'm a 39-year-old married engineer with a bachelor's, 45 hrs/week."
         │
         ▼
┌──────────────────────┐   tool schema built from the FITTED encoder,
│  Claude (parsing)    │   so categories can never drift from training
│  src/llm_interface   │
└──────────┬───────────┘
           │ tool_use: {age: 39, education: "Bachelors", ...}
           ▼
┌──────────────────────┐
│  sklearn Pipeline    │   impute → scale/one-hot → XGBoost
│  models/best_model   │   (preprocessing is INSIDE the artifact)
└──────────┬───────────┘
           │ tool_result: {">50K", p=0.72}
           ▼
┌──────────────────────┐
│  Claude (explaining) │ → "The model predicts over $50K, at 72% confidence.
│  src/app.py (UI)     │    The biggest factors were... Note this is 1994 data."
└──────────────────────┘
```

Three decisions worth calling out:

1. **Preprocessing lives inside the model artifact.** The `ColumnTransformer`
   and the classifier are one `Pipeline`, fitted together on the training fold
   only. That prevents test-set leakage *and* means the app feeds raw
   user-shaped values straight in — the serving path cannot encode differently
   than training did.
2. **The tool schema is derived from the fitted pipeline**, not hardcoded. The
   allowed categories come from the `OneHotEncoder`'s learned vocabulary and the
   defaults from the `SimpleImputer`'s learned statistics. Retrain on different
   data and the LLM's schema follows automatically.
3. **The LLM never produces a number.** It decides *whether* it has enough
   information, maps words to categories, and explains. Every probability comes
   from `model.predict_proba`.

---

## Repository layout

```
├── configs/config.yaml       # All hyperparameters — nothing hardcoded in train.py
├── src/
│   ├── preprocess.py         # Loading, cleaning, the feature pipeline
│   ├── train.py              # Trains each config, logs to MLflow
│   ├── evaluate.py           # Metrics + mlflow.search_runs() comparison
│   ├── llm_interface.py      # Tool schema, parsing, tool-use loop (no UI)
│   └── app.py                # Streamlit chat UI
├── tests/                    # 15 tests: preprocessing, model, interface
├── data/                     # Dataset (gitignored)
├── models/                   # Exported best model (gitignored)
└── Dockerfile
```

---

## Setup

**1. Environment** (Python 3.13 — 3.14 lacks reliable mlflow/xgboost wheels):

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

**2. Get the data.** Download the [UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult)
and put `adult.data` and `adult.test` in `data/`. The loader concatenates both
into the full 48,842 rows and does its own stratified split.

**3. API key.** Copy `.env.example` to `.env` and add your Anthropic key. The
code reads it from the environment — it is never hardcoded, and `.env` is
gitignored.

```bash
cp .env.example .env
```

---

## Usage

Train all six configurations, then pick and export the winner:

```bash
python -m src.train
```

```bash
python -m src.evaluate
```

`evaluate` prints a leaderboard, uses `mlflow.search_runs()` to rank runs by F1,
and writes the best pipeline to `models/best_model.joblib`.

Launch the app:

```bash
streamlit run src/app.py
```

Browse the experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run the tests:

```bash
pytest tests/ -v
```

With Docker (after training once, so `models/` is populated):

```bash
docker build -t income-predictor . && docker run -p 8501:8501 --env-file .env income-predictor
```

---

## Results

Six configurations, all evaluated on the same held-out 20% split (9,758 rows):

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **xgboost_tuned** | **0.8797** | 0.7865 | 0.6828 | **0.7310** | **0.9330** |
| gradient_boosting | 0.8763 | 0.7923 | 0.6550 | 0.7171 | 0.9299 |
| random_forest_deep | 0.8693 | 0.7907 | 0.6177 | 0.6936 | 0.9217 |
| logistic_regression_balanced | 0.8061 | 0.5627 | 0.8532 | 0.6781 | 0.9075 |
| logistic_regression_baseline | 0.8508 | 0.7266 | 0.6040 | 0.6597 | 0.9079 |
| random_forest_shallow | 0.8606 | 0.8177 | 0.5377 | 0.6488 | 0.9120 |

**Why XGBoost.** It leads on F1 and ROC-AUC and is within 0.003 of the best
precision. Selection is on **F1, not accuracy**, because the target is
imbalanced — only 23.9% earn >$50K, so a model predicting "everyone earns less"
scores 76% accuracy while being useless. F1 forces the positive class to
actually be found.

The comparison is instructive: `random_forest_shallow` has the *highest*
precision (0.818) but the worst F1, because it only catches 54% of high earners.
`logistic_regression_balanced` is the mirror image — class weighting pushes
recall to 0.853 at the cost of precision collapsing to 0.563. XGBoost is the
only configuration that gets both above 0.68.

**Interesting findings:**

- **Deduplication order changes the dataset size by 13%.** Removing duplicates
  *after* dropping `fnlwgt` deleted 6,374 rows; doing it before deletes 52. Two
  different census respondents can share every remaining demographic field —
  they are not duplicates, and treating them as such throws away real data.
- **`fnlwgt` and `education-num` were dropped deliberately.** `fnlwgt` is a
  census sampling weight, a property of the survey rather than the person.
  `education-num` is an exact ordinal restatement of `education`, so keeping
  both double-counts one signal.
- **Recall is the hard part.** Every model's precision beats its recall. High
  earners are heterogeneous; low earners are easier to characterize.

---

## Reflection


**What I learned.** The most valuable idea was making the tool schema a
*derivative* of the fitted model rather than a parallel hand-maintained
definition. My first instinct was to hardcode the category lists in the prompt,
which works right up until the model is retrained on different data and the LLM
starts confidently sending categories the encoder has never seen. Reading them
off `OneHotEncoder.categories_` makes that class of bug structurally impossible.

**What was challenging.** Two things. First, the leakage question is subtler
than "fit on train, transform on test" — it's about where the fitted state
*lives*. Putting the transformer in the same `Pipeline` as the classifier solved
correctness and the serving-skew problem at once. Second, edge-case handling
turned out to be a prompt-design problem more than a code problem: the model
would try to be helpful and predict from two facts. The fix was making the
system prompt name the required fields explicitly, plus a defensive
`missing_required()` check so a premature tool call returns an error to the LLM
rather than a prediction built from defaults.

**What I'd improve with more time.**

- **Threshold tuning.** I selected on F1 at the default 0.5 cutoff. Sweeping the
  decision threshold would let the precision/recall balance be chosen for a
  specific use case rather than left at the default.
- **Fairness audit.** The data encodes 1994 income patterns by sex and race, and
  the model reproduces them. I disclose this in the UI, but I did not measure
  it — computing per-group error rates is the honest next step.
- **Calibration.** The app shows a probability to one decimal place. I have not
  verified those probabilities are calibrated, so "72%" may not mean what a user
  reasonably assumes it means.

---

## Limitations

The model is trained on 1994 US census data. The $50,000 threshold is not
inflation-adjusted, and the social and economic patterns in the data are three
decades old. It reflects the biases of that dataset. This is a demonstration
project, not financial advice.
