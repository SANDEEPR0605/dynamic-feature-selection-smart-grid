"""
src/online_learner.py

Fully incremental online learning benchmark for the "Dynamic Feature
Selection for Non-Stationary Environments" dissertation project.

Implements Chapter 3, Section 3.9 and Chapter 4, Section 4.10: a
genuinely incremental River regressor that updates continuously as each
new observation arrives, without waiting for a drift signal or a full
retrain -- included as a reference point for how far drift-triggered
retraining of batch models (Phases 6-7) goes relative to a model that is
always updating. Two candidates are provided, consistent with Chapter 3:
an online Lasso-style linear model (L1-regularised) and River's
Hoeffding Tree Regressor. Chapter 4 specifies the choice between them is
made empirically ("selected based on which performs more reliably in
preliminary testing") rather than fixed in advance -- see
`select_online_model` below.

Unlike the drift-triggered framework (Phases 6-7), this learner does not
perform any feature selection: it uses the full, fixed feature set
throughout, since it is testing a different axis entirely (continuous
parameter updating vs static/dynamic feature selection), not competing
on the feature-selection dimension itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from river import linear_model, optim, preprocessing, tree

import metrics as mt

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Candidate online model registry
# ---------------------------------------------------------------------------

def make_online_lasso(l1: float = 0.01, l2: float = 0.0):
    """An online Lasso-style linear model: L1-regularised linear
    regression with per-feature standardisation (both incremental,
    updated online -- necessary here since, unlike the batch models in
    Phases 2-7, this model is sensitive to feature scale and cannot be
    fit-then-frozen on a training set upfront).
    """
    return preprocessing.StandardScaler() | linear_model.LinearRegression(l1=l1, l2=l2)


def make_hoeffding_tree(grace_period: int = 200, max_depth=None):
    """River's Hoeffding Tree Regressor -- an incremental decision tree
    that splits nodes once enough statistical evidence has accumulated.
    Scale-invariant, so no preprocessing is needed.
    """
    return tree.HoeffdingTreeRegressor(grace_period=grace_period, max_depth=max_depth)


ONLINE_MODEL_REGISTRY = {
    "online_lasso": make_online_lasso,
    "hoeffding_tree": make_hoeffding_tree,
}

ONLINE_MODEL_DISPLAY_NAMES = {
    "online_lasso": "Online Lasso (L1-regularised)",
    "hoeffding_tree": "Hoeffding Tree Regressor",
}


def get_online_model(name: str):
    if name not in ONLINE_MODEL_REGISTRY:
        raise KeyError(f"Unknown online model '{name}'. Options: {list(ONLINE_MODEL_REGISTRY)}")
    return ONLINE_MODEL_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Running an online model: predict-then-learn (test-then-train)
# ---------------------------------------------------------------------------

@dataclass
class OnlineRunResult:
    predictions: np.ndarray
    errors: np.ndarray
    y_true: np.ndarray
    total_runtime_s: float = 0.0
    model_name: str = ""


def run_online_learner(model_name: str, stream_df: pd.DataFrame, feature_cols: List[str], target_col: str) -> OnlineRunResult:
    """Process `stream_df` one row at a time using the standard online-
    learning predict-then-learn pattern: predict on the row BEFORE
    updating the model with its true value, so predictions never benefit
    from having already seen the answer -- then learn_one updates the
    model incrementally, immediately, no batching or retraining.
    """
    model = get_online_model(model_name)
    n = len(stream_df)
    predictions = np.empty(n)
    y_true = stream_df[target_col].to_numpy()

    records = stream_df[feature_cols].to_dict("records")

    with mt.timer() as t:
        for i, x in enumerate(records):
            pred = model.predict_one(x)
            predictions[i] = pred if pred is not None else 0.0
            model.learn_one(x, y_true[i])

    errors = np.abs(y_true - predictions)
    return OnlineRunResult(
        predictions=predictions, errors=errors, y_true=y_true,
        total_runtime_s=t["seconds"], model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Preliminary model selection (Chapter 4, Section 4.10)
# ---------------------------------------------------------------------------

def select_online_model(stream_df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[str, pd.DataFrame]:
    """Run both candidate online models on `stream_df` and return the name
    of whichever achieves lower MAE, along with a comparison table -- the
    empirical selection Chapter 4 specifies ("selected based on which
    performs more reliably in preliminary testing").
    """
    rows = []
    for name in ONLINE_MODEL_REGISTRY:
        result = run_online_learner(name, stream_df, feature_cols, target_col)
        acc = mt.compute_predictive_metrics(result.y_true, result.predictions, n_features=len(feature_cols))
        rows.append({
            "model": name,
            "display_name": ONLINE_MODEL_DISPLAY_NAMES[name],
            **acc,
            "runtime_s": result.total_runtime_s,
        })

    comparison_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    selected_name = comparison_df.iloc[0]["model"]
    return selected_name, comparison_df


# ---------------------------------------------------------------------------
# Summarising an OnlineRunResult with the full Chapter 3, Table 3.5 metric
# suite, in the same shape as dynamic_framework.summarize_result so the
# two are directly comparable in Phase 9.
# ---------------------------------------------------------------------------

def summarize_result(result: OnlineRunResult, n_features: int, penalty_alpha: float = 1.5) -> Dict[str, float]:
    acc = mt.compute_predictive_metrics(result.y_true, result.predictions, n_features=n_features)
    return {
        **acc,
        "Penalised_MAE": mt.penalised_mae(result.y_true, result.predictions, alpha=penalty_alpha),
        "Penalised_Loss": mt.penalised_loss(result.y_true, result.predictions, alpha=penalty_alpha),
        "total_runtime_s": result.total_runtime_s,
        "selection_runtime_s": 0.0,  # no feature-selection step in this benchmark
        "n_drift_events": 0,          # no drift detection in this benchmark
        "n_retrains": 0,              # continuous updates, not discrete retrains
        "mean_feature_stability_index": 1.0,  # feature set never changes
        "mean_importance_consistency": float("nan"),  # no importance ranking computed
        "feature_switch_rate": 0.0,
        "mean_adaptation_time_rows": float("nan"),
        "n_adaptation_events_recovered": 0,
    }
