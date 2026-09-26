# Capstone Project 01 — Train a Machine Learning Model (End-to-End)

> **Deliverable:** A trained, evaluated, versioned ML model — packaged with the data pipeline, feature engineering code, cross-validation strategy, and a serving endpoint — for a concrete prediction task. The domain is deliberately practical: **predicting the outcome of a soccer match** given pre-match features.
>
> **Time:** 6–10 hours the first time, 2–3 hours on repeat.
>
> **What you'll be able to say afterwards:** "I built an ML classifier that reaches X% accuracy on Y task. Here's the CV strategy, the leakage checks I ran, and why I picked model M over N. I versioned it with MLflow and served it behind a FastAPI endpoint."

---

## 1. Project Overview

We will predict the outcome of a soccer match — home win, draw, away win — from pre-match features only (no in-game data). This is a **three-class classification** task with well-known baselines: the bookmaker odds imply roughly 50–55% accuracy for the best public models on English Premier League data.

Why this task:
- Real-world messiness: leakage, temporal splits, class imbalance
- Requires all the pieces of a real ML pipeline (ingest, features, training, eval, packaging)
- Extensible into a live prediction service
- Fits Brian's soccer analytics platform interest

### Architecture
![IMG-CAP01-01](/7%20-%20Projects/images/IMG-CAP01-01.jpg)


### Prerequisites

- Python 3.11+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `mlflow`, `fastapi`, `uvicorn`, `pydantic>=2`, `httpx`, `pytest`
- 500 MB free disk
- An account on football-data.co.uk (free, no API key needed for CSVs)

### Repo scaffold

```
soccer-predictor/
├── pyproject.toml
├── data/
│   ├── raw/          # Downloaded CSVs
│   └── processed/    # Cleaned parquet files
├── src/
│   ├── __init__.py
│   ├── ingest.py     # Fetch and clean
│   ├── features.py   # Feature engineering
│   ├── train.py      # Training entrypoint
│   ├── evaluate.py   # Metrics & diagnostics
│   └── serve.py      # FastAPI service
├── tests/
│   └── test_features.py
├── models/           # Local model artifacts (also in MLflow)
├── notebooks/
│   └── 01-explore.ipynb
└── README.md
```

---

## 2. Step-by-Step Implementation

### Step 1 — Data ingestion (`src/ingest.py`)

**Why:** ML is a data problem before it's an algorithm problem. Fetch multiple seasons; the model needs history to learn team quality.

```python
"""ingest.py — download and normalize match data."""
from pathlib import Path
import pandas as pd
import httpx

RAW_DIR = Path("data/raw")
SEASONS = ["2223", "2324", "2425", "2526"]  # last few seasons
LEAGUE = "E0"  # Premier League code on football-data.co.uk
BASE_URL = "https://www.football-data.co.uk/mmz4281"

def fetch_season(season: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{season}/{LEAGUE}.csv"
    out = RAW_DIR / f"{LEAGUE}_{season}.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        r = httpx.get(url, timeout=30, follow_redirects=True)
        r.raise_for_status()
        out.write_bytes(r.content)
    return pd.read_csv(out)

def load_all() -> pd.DataFrame:
    frames = [fetch_season(s) for s in SEASONS]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = load_all()
    print(f"Loaded {len(df)} matches from {df.Date.min()} to {df.Date.max()}")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/processed/matches.parquet")
```

**Why parquet?** Columnar, typed, ~5× smaller than CSV, and preserves datetime types.

### Step 2 — Feature engineering (`src/features.py`)

**The core insight of soccer prediction:** rolling recent form and team strength matter more than any single-match feature. The trap: **temporal leakage** — computing a feature using a stat that hadn't been observed by match time.

```python
"""features.py — leakage-safe feature construction."""
import pandas as pd
import numpy as np

def add_rolling_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """For each match, compute the home/away teams' rolling stats over their
    last n matches BEFORE the current match — never including it."""
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Build a long form: one row per (match, team, is_home)
    long = pd.concat([
        df.rename(columns={"HomeTeam": "team", "AwayTeam": "opp",
                            "FTHG": "gf", "FTAG": "ga"}).assign(is_home=1),
        df.rename(columns={"AwayTeam": "team", "HomeTeam": "opp",
                            "FTAG": "gf", "FTHG": "ga"}).assign(is_home=0),
    ], ignore_index=True).sort_values(["team", "Date"])

    # Points earned in each match (from the team's perspective)
    long["points"] = np.where(
        long["gf"] > long["ga"], 3,
        np.where(long["gf"] == long["ga"], 1, 0)
    )
    # Rolling — SHIFTED so we don't include the current match's outcome
    for col in ["gf", "ga", "points"]:
        long[f"roll_{col}_{n}"] = (
            long.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        )

    # Rest days since last match
    long["rest_days"] = long.groupby("team")["Date"].diff().dt.days

    # Merge home/away versions back onto match rows
    home_feats = long[long.is_home == 1].rename(columns={
        "team": "HomeTeam",
        **{f"roll_{c}_{n}": f"h_roll_{c}_{n}" for c in ["gf", "ga", "points"]},
        "rest_days": "h_rest",
    })[["Date", "HomeTeam"] + [f"h_roll_{c}_{n}" for c in ["gf","ga","points"]] + ["h_rest"]]

    away_feats = long[long.is_home == 0].rename(columns={
        "team": "AwayTeam",
        **{f"roll_{c}_{n}": f"a_roll_{c}_{n}" for c in ["gf", "ga", "points"]},
        "rest_days": "a_rest",
    })[["Date", "AwayTeam"] + [f"a_roll_{c}_{n}" for c in ["gf","ga","points"]] + ["a_rest"]]

    df = df.merge(home_feats, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_feats, on=["Date", "AwayTeam"], how="left")
    return df

def add_elo(df: pd.DataFrame, k: float = 20, home_adv: float = 60) -> pd.DataFrame:
    """Elo ratings updated match by match. Feature value is Elo BEFORE the match."""
    df = df.copy().sort_values("Date").reset_index(drop=True)
    elo = {}
    h_elo, a_elo = [], []
    for _, row in df.iterrows():
        h, a = row.HomeTeam, row.AwayTeam
        rh = elo.get(h, 1500)
        ra = elo.get(a, 1500)
        h_elo.append(rh)
        a_elo.append(ra)
        # Update after the match
        eh = 1 / (1 + 10 ** ((ra - rh - home_adv) / 400))
        sh = 1.0 if row.FTHG > row.FTAG else (0.5 if row.FTHG == row.FTAG else 0.0)
        elo[h] = rh + k * (sh - eh)
        elo[a] = ra + k * ((1 - sh) - (1 - eh))
    df["h_elo"] = h_elo
    df["a_elo"] = a_elo
    df["elo_diff"] = df["h_elo"] - df["a_elo"]
    return df

def target(df: pd.DataFrame) -> pd.DataFrame:
    """FTR values: 'H' home win, 'D' draw, 'A' away win. Encode as 0/1/2."""
    df = df.copy()
    df["y"] = df["FTR"].map({"H": 0, "D": 1, "A": 2}).astype("int8")
    return df

FEATURE_COLS = [
    "h_roll_gf_5", "h_roll_ga_5", "h_roll_points_5", "h_rest",
    "a_roll_gf_5", "a_roll_ga_5", "a_roll_points_5", "a_rest",
    "h_elo", "a_elo", "elo_diff",
]

def build(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rolling_form(df)
    df = add_elo(df)
    df = target(df)
    # Drop early rows with insufficient history
    df = df.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)
    return df
```

**Why shifted rolling?** If you compute a rolling mean including the current match, you leak that match's result into its own feature — the model learns to "predict" the label from a feature that already contains it. Almost all suspiciously-high validation scores in beginner ML projects come from a leakage bug like this.

### Step 3 — Time-based splits (`src/train.py` — part 1)

**Why:** IID random splits would train on future matches and test on past — the opposite of production. Use **time-based** splits.

```python
"""train.py — end-to-end training with time splits + CV."""
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier

from src.features import build, FEATURE_COLS

def load() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/matches.parquet")
    return build(df)

def time_split(df: pd.DataFrame, holdout_frac: float = 0.15):
    """Last `holdout_frac` of matches (by date) is the test set."""
    df = df.sort_values("Date").reset_index(drop=True)
    cutoff = int(len(df) * (1 - holdout_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]

def rolling_time_cv(df: pd.DataFrame, n_folds: int = 5):
    """Yield (train_idx, val_idx) tuples where val is always after train."""
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train_end = i * fold_size
        val_end = train_end + fold_size
        yield np.arange(0, train_end), np.arange(train_end, min(val_end, n))
```

### Step 4 — Models & MLflow (`src/train.py` — part 2)

```python
def build_models():
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial")),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=8,
                                      n_jobs=-1, random_state=42),
        "xgb": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        ),
    }

def train_and_track(name, model, df):
    mlflow.set_experiment("soccer-outcome")
    with mlflow.start_run(run_name=name):
        # Rolling-time CV
        cv_scores = []
        for fold, (tr, va) in enumerate(rolling_time_cv(df)):
            X_tr, y_tr = df.iloc[tr][FEATURE_COLS], df.iloc[tr]["y"]
            X_va, y_va = df.iloc[va][FEATURE_COLS], df.iloc[va]["y"]
            m = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_va)
            cv_scores.append({
                "fold": fold,
                "accuracy": accuracy_score(y_va, p.argmax(1)),
                "log_loss": log_loss(y_va, p, labels=[0,1,2]),
            })
            mlflow.log_metrics({f"cv_acc_fold{fold}": cv_scores[-1]["accuracy"],
                                f"cv_logloss_fold{fold}": cv_scores[-1]["log_loss"]})

        cv_df = pd.DataFrame(cv_scores)
        mlflow.log_metric("cv_acc_mean", cv_df.accuracy.mean())
        mlflow.log_metric("cv_logloss_mean", cv_df.log_loss.mean())

        # Fit on train, evaluate on holdout
        train_df, test_df = time_split(df)
        model.fit(train_df[FEATURE_COLS], train_df["y"])
        proba = model.predict_proba(test_df[FEATURE_COLS])
        pred = proba.argmax(1)

        acc = accuracy_score(test_df["y"], pred)
        ll = log_loss(test_df["y"], proba, labels=[0,1,2])
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_log_loss", ll)

        # Baseline: always predict home win (majority class in most leagues)
        baseline_acc = (test_df["y"] == 0).mean()
        mlflow.log_metric("baseline_home_accuracy", baseline_acc)

        # Log params, model
        params = model.get_params() if hasattr(model, "get_params") else {}
        mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
        mlflow.sklearn.log_model(model, "model")

        print(f"{name}: acc={acc:.3f} ll={ll:.3f} (baseline={baseline_acc:.3f})")
        return {"name": name, "acc": acc, "log_loss": ll, "model": model}

def main():
    df = load()
    print(f"Training on {len(df)} matches; features: {FEATURE_COLS}")
    models = build_models()
    results = [train_and_track(name, m, df) for name, m in models.items()]
    best = min(results, key=lambda r: r["log_loss"])
    print(f"\nBest model: {best['name']} (log-loss {best['log_loss']:.3f})")

if __name__ == "__main__":
    main()
```

**Why log-loss over accuracy?** Accuracy hides calibration. A model that says "70% home win" for every match and gets 55% accuracy can beat a model that hits 60% accuracy but confidently outputs 95% wrong. Betting-adjacent tasks care about probability quality; log-loss (aka cross-entropy) is the correct proper scoring rule.

### Step 5 — Diagnostics (`src/evaluate.py`)

```python
"""evaluate.py — diagnostics beyond accuracy."""
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report

def calibration_report(y_true, proba, class_idx: int, n_bins: int = 10):
    """For a given class, how well-calibrated are the predicted probabilities?"""
    y_bin = (y_true == class_idx).astype(int)
    prob_class = proba[:, class_idx]
    frac_pos, mean_pred = calibration_curve(y_bin, prob_class, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})

def per_class_metrics(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=["Home","Draw","Away"], output_dict=True)

def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0,1,2])
```

`[IMG-CAP01-02]` — *Prompt: A 2×2 grid of diagnostic plots for a 3-class soccer outcome classifier. Top-left: confusion matrix as a heatmap, rows/columns labeled Home/Draw/Away, cells with counts and color intensity. Top-right: calibration plot — one line per class showing mean predicted probability vs actual frequency, with the y=x diagonal as ideal. Bottom-left: log-loss over 5 CV folds as a bar chart. Bottom-right: bar chart of accuracy for three models (LogReg / RF / XGB) with baseline "always home" as a dashed reference line. Clean matplotlib/seaborn styling, grid, legends.*

### Step 6 — Model registry promotion

```python
# after training, promote the best model to the champion alias
import mlflow
client = mlflow.tracking.MlflowClient()

# Register the best run's model
run_id = "abcd1234..."  # get from the training output or MLflow UI
mv = mlflow.register_model(f"runs:/{run_id}/model", "soccer-outcome-classifier")
client.set_registered_model_alias(
    name="soccer-outcome-classifier",
    alias="champion",
    version=mv.version,
)
```

### Step 7 — Serving (`src/serve.py`)

```python
"""serve.py — FastAPI endpoint loading the champion model at startup."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import mlflow.pyfunc

MODEL_URI = "models:/soccer-outcome-classifier@champion"
STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    yield
    STATE.clear()

app = FastAPI(lifespan=lifespan)

class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    h_roll_gf_5: float
    h_roll_ga_5: float
    h_roll_points_5: float
    h_rest: float
    a_roll_gf_5: float
    a_roll_ga_5: float
    a_roll_points_5: float
    a_rest: float
    h_elo: float
    a_elo: float
    elo_diff: float

class PredictionResponse(BaseModel):
    home_win: float
    draw: float
    away_win: float
    predicted: str

@app.post("/predict", response_model=PredictionResponse)
def predict(req: MatchRequest):
    import pandas as pd
    X = pd.DataFrame([req.model_dump()])
    proba = STATE["model"].predict(X)[0]  # returns shape (3,)
    labels = ["Home", "Draw", "Away"]
    return PredictionResponse(
        home_win=float(proba[0]),
        draw=float(proba[1]),
        away_win=float(proba[2]),
        predicted=labels[int(proba.argmax())],
    )

@app.get("/healthz")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}
```

Run: `uvicorn src.serve:app --port 8000`. Test:
```bash
curl -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{
  "h_roll_gf_5": 1.8, "h_roll_ga_5": 0.6, "h_roll_points_5": 2.4, "h_rest": 4,
  "a_roll_gf_5": 1.0, "a_roll_ga_5": 1.2, "a_roll_points_5": 1.2, "a_rest": 5,
  "h_elo": 1650, "a_elo": 1480, "elo_diff": 170
}'
```

### Step 8 — Tests (`tests/test_features.py`)

```python
import pandas as pd
import numpy as np
from src.features import add_rolling_form, add_elo, FEATURE_COLS

def make_fake_matches():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-08-01", "2024-08-10", "2024-08-17"]),
        "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
        "AwayTeam": ["Chelsea", "Arsenal", "Chelsea"],
        "FTHG": [2, 0, 1],
        "FTAG": [1, 1, 1],
        "FTR": ["H", "A", "D"],
    })

def test_no_leakage_in_rolling():
    df = make_fake_matches()
    out = add_rolling_form(df, n=2)
    # First match: no history for either team → NaN rolling stats
    assert pd.isna(out.iloc[0]["h_roll_gf_5"])
    # Second match: Chelsea (home) has one prior game
    assert not pd.isna(out.iloc[1]["h_roll_gf_5"])

def test_elo_updates_after_match():
    df = make_fake_matches()
    out = add_elo(df)
    # Elo BEFORE any match = 1500 for both
    assert out.iloc[0]["h_elo"] == 1500
    # After Arsenal beats Chelsea 2-1 at home, Arsenal's elo > Chelsea's
    # (visible in row 1 where Chelsea is home and Arsenal is away)
    assert out.iloc[1]["a_elo"] > out.iloc[1]["h_elo"]
```

Run: `pytest tests/ -v`

---

## 3. Why Each Decision Matters (Interview-Ready)

| Decision | Rationale |
|----------|-----------|
| **Time-based splits** | Random splits leak future info to the past; production predicts forward in time |
| **Shifted rolling means** | Non-shifted rolling means include the current match; the model "cheats" |
| **Log-loss primary, accuracy secondary** | Calibrated probabilities are what downstream (betting, decisions) needs |
| **Elo as a feature** | Encodes team strength across time; the single strongest baseline for soccer |
| **Three models tried** | Bias-variance sanity check: linear (LogReg) vs low-bias tree (RF) vs boosted (XGB) |
| **MLflow versioning + alias** | Reproducibility + zero-downtime champion promotion |
| **Pydantic `extra="forbid"`** | Rejects unknown feature names silently entering the request |
| **FastAPI `lifespan` load** | Model loaded once at startup, not per request (100× latency saved) |

---

## 4. Expected Results

Realistic outcomes on this dataset:
- LogReg: ~50–52% accuracy, log-loss ~1.03
- RF: ~52–54% accuracy, log-loss ~1.00
- XGB: ~53–55% accuracy, log-loss ~0.97
- Baseline (always home): ~44% accuracy, log-loss ~1.10

**If your numbers are wildly higher, suspect leakage.** Common bugs: including future info in rolling stats, using the season's full-year table as a feature, encoding target-leakage stats like "goals scored this season."

---

## 5. Extensions

- Add features: head-to-head win rate, days since manager change, injury count (needs another data source)
- Try a small neural net (MLP) — likely no improvement on this data volume, useful negative result
- Bet-sizing wrapper: convert probabilities + bookmaker odds into Kelly-optimal stake sizes
- Retrain job: nightly, keyed by "latest match date"; auto-promote if log-loss improves > 1%

---

## 6. Interview Talking Points

If asked "walk me through an ML project you built":

1. **Frame the problem.** "3-class classification, probabilistic output required for downstream utility."
2. **The data story.** "Public CSVs, ~1500 matches, sorted by date. First cleaning step was normalizing team-name spellings across seasons."
3. **The leakage lesson.** "My initial version had non-shifted rolling means; validation accuracy hit 71% which was a red flag. Rebuilt the feature step with `shift(1)` before `rolling()`; accuracy dropped to a realistic 53%."
4. **Why log-loss.** "Downstream users need calibrated probabilities, not just argmax. I evaluated on both."
5. **Model selection.** "Three families tried under identical CV; XGBoost narrowly won on log-loss. LogReg is close and easier to explain."
6. **Serving.** "MLflow champion alias, FastAPI loads at startup, ~15ms per prediction. Rolling window features are computed offline and passed in."
7. **What I'd do next.** "Injury feature is the biggest missing signal; also want a live-odds feature and a bet-sizing wrapper."

---

## 7. References

- Constantinou & Fenton, *Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models*
- David Sumpter, *Soccermatics* (statistical soccer analytics)
- MLflow tutorials: https://mlflow.org/docs/latest/tutorials-and-examples/
- football-data.co.uk data dictionary
- Wojtek Kowalczyk et al. — historical papers on Elo-based ranking systems in team sports
