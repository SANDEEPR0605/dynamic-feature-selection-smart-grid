"""
src/models.py

Candidate regression models for the "Dynamic Feature Selection for
Non-Stationary Environments" dissertation project.

Implements Chapter 3, Section 3.3 and Chapter 4, Section 4.4: a common
fit/predict interface wrapping Linear Regression, Random Forest, and
XGBoost, so that the same code path trains and evaluates all three
models here in Phase 2, and is reused unchanged inside the integrated
dynamic framework in Phase 7 (Chapter 4, Section 4.2).

Random seeds are fixed throughout for reproducibility -- every score
reported in Chapter 5 should be reproducible from this module and the
processed data alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import metrics as mt

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model registry -- the single source of truth for "what is a candidate
# model" (Chapter 3, Section 3.3). Later phases (5, 6, 7) look up models
# by these same names.
# ---------------------------------------------------------------------------

def _make_linear_regression() -> LinearRegression:
    return LinearRegression()


def _make_random_forest() -> RandomForestRegressor:
    # max_depth and min_samples_leaf are capped deliberately: with
    # max_depth=None on ~40,000 training rows, trees grow deep enough that
    # the serialised model reaches several hundred MB, which is impractical
    # to persist/version across Colab sessions and later phases (4-7) for
    # negligible accuracy gain over a depth-capped forest.
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _make_xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


MODEL_REGISTRY = {
    "linear_regression": _make_linear_regression,
    "random_forest": _make_random_forest,
    "xgboost": _make_xgboost,
}

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def get_model(name: str):
    """Construct a fresh, unfitted instance of the named candidate model."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Fit / predict / evaluate
# ---------------------------------------------------------------------------

def fit_model(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a fresh instance of the named model and return it, along with
    the fit runtime in seconds.
    """
    model = get_model(name)
    with mt.timer() as t:
        model.fit(X_train, y_train)
    return model, t["seconds"]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], float]:
    """Predict on X_test and compute the predictive-accuracy metric suite.

    Returns (metrics_dict, predict_runtime_seconds).
    """
    with mt.timer() as t:
        y_pred = model.predict(X_test)
    scores = mt.compute_predictive_metrics(y_test, y_pred, n_features=X_test.shape[1])
    return scores, t["seconds"]


def compare_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train and evaluate every candidate model (Chapter 3, Section 3.3).

    Returns:
        results_df: one row per model, with predictive metrics and runtimes.
        fitted_models: dict of {model_name: fitted_model_object}, so the
            caller can persist them (see save_model) without refitting.
    """
    rows = []
    fitted_models = {}

    for name in MODEL_REGISTRY:
        model, fit_seconds = fit_model(name, X_train, y_train)
        scores, predict_seconds = evaluate_model(model, X_test, y_test)

        row = {
            "model": name,
            "display_name": MODEL_DISPLAY_NAMES[name],
            **scores,
            "fit_runtime_s": fit_seconds,
            "predict_runtime_s": predict_seconds,
        }
        rows.append(row)
        fitted_models[name] = model

    results_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return results_df, fitted_models


# ---------------------------------------------------------------------------
# Ranking and artefact selection (Chapter 4, Section 4.4: "the
# best-performing model -- and, separately, the identity of the non-best
# model ... -- are recorded as artefacts consumed by every later phase")
# ---------------------------------------------------------------------------

def rank_models(results_df: pd.DataFrame, by: str = "MAE") -> List[str]:
    """Return model names ranked best-to-worst by the given metric
    (ascending -- lower is better for MAE/RMSE/MAPE).
    """
    return results_df.sort_values(by)["model"].tolist()


# ---------------------------------------------------------------------------
# Persistence -- later phases (4, 5, 6, 7) load the best model back in
# rather than refitting it.
# ---------------------------------------------------------------------------

def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)


def save_artifacts(artifacts: dict, path: str | Path) -> None:
    """Save small JSON artefacts (best model name, rankings, etc.)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifacts, f, indent=2)


def load_artifacts(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
