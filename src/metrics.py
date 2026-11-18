"""
src/metrics.py

Shared evaluation metrics for the "Dynamic Feature Selection for
Non-Stationary Environments" dissertation project.

Implements the full metric suite defined in Chapter 3, Section 3.10
(Table 3.5): predictive accuracy, penalised (risk-aware) error,
adaptation, and interpretability metrics. Written once here so that
every phase (2, 4, 5, 7, 8, 9) uses exactly the same formulas -- Chapter
5's tables depend on that consistency.

Only predictive-accuracy metrics (mae, rmse, r2, adjusted_r2, mape,
compute_predictive_metrics) are exercised in Phase 2. The remaining
functions are used from Phase 4 onward and are included here now so the
module has one stable home.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Predictive accuracy metrics (Chapter 3, Table 3.5) -- used from Phase 2
# ---------------------------------------------------------------------------

def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination, R^2 = 1 - SS_res / SS_tot."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def adjusted_r2(y_true, y_pred, n_features: int) -> float:
    """R^2 adjusted for the number of predictors, n_features (p).

    Useful once the active feature-subset size changes across windows
    (Chapter 3, Table 3.5), since plain R^2 does not penalise adding
    features that don't genuinely help.
    """
    n = len(np.asarray(y_true))
    r2_val = r2(y_true, y_pred)
    denom = n - n_features - 1
    if denom <= 0 or np.isnan(r2_val):
        return float("nan")
    return float(1 - (1 - r2_val) * (n - 1) / denom)


def mape(y_true, y_pred, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error, as a percentage.

    `epsilon` guards against division by (near-)zero target values --
    the dataset's minimum Predicted Load is small but non-zero, so this
    mainly protects against degenerate windows during streaming phases.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_predictive_metrics(y_true, y_pred, n_features: Optional[int] = None) -> Dict[str, float]:
    """Bundle MAE, RMSE, R^2, Adjusted R^2 (if n_features given), and MAPE.

    This is the single function every phase should call so that Chapter
    5's tables are always built from identical metric definitions.
    """
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    if n_features is not None:
        out["Adjusted_R2"] = adjusted_r2(y_true, y_pred, n_features)
    return out


# ---------------------------------------------------------------------------
# Penalised (risk-aware) error (Chapter 3, Section 3.10)
# ---------------------------------------------------------------------------

def penalised_mae(y_true, y_pred, alpha: float = 1.5) -> float:
    """Asymmetric MAE that weights under-prediction (y_pred < y_true) by
    `alpha` (> 1), since under-forecasting grid load is operationally
    costlier than over-forecasting it.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    weights = np.where(y_pred < y_true, alpha, 1.0)
    return float(np.mean(weights * errors))


def penalised_loss(y_true, y_pred, alpha: float = 1.5) -> float:
    """Asymmetric squared penalised loss -- penalises large
    under-predictions super-linearly (Chapter 3, Section 3.10).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    under = np.maximum(y_true - y_pred, 0) ** 2
    over = np.maximum(y_pred - y_true, 0) ** 2
    return float(np.mean(alpha * under + over))


# ---------------------------------------------------------------------------
# Adaptation metrics (Chapter 3, Section 3.10) -- used from Phase 7
# ---------------------------------------------------------------------------

def adaptation_time(
    window_errors: Sequence[float],
    drift_window_idx: int,
    pre_drift_error: float,
    tolerance: float = 1.05,
) -> Optional[int]:
    """Number of windows after a detected drift point until error returns
    to at or below `pre_drift_error * tolerance`.

    Returns None if the stream ends before recovery is observed (this is
    itself a meaningful result, reported as-is in Chapter 5 rather than
    silently coerced to a number).
    """
    threshold = pre_drift_error * tolerance
    for offset, err in enumerate(window_errors[drift_window_idx:]):
        if err <= threshold:
            return offset
    return None


# ---------------------------------------------------------------------------
# Interpretability metrics (Chapter 3, Section 3.10) -- used from Phase 5
# ---------------------------------------------------------------------------

def feature_stability_index(features_t: Sequence[str], features_t_minus_1: Sequence[str]) -> float:
    """Jaccard similarity between two consecutive windows' selected
    feature sets: |F_t intersect F_t-1| / |F_t union F_t-1|.
    """
    set_t, set_t1 = set(features_t), set(features_t_minus_1)
    if not set_t and not set_t1:
        return 1.0
    union = set_t | set_t1
    if not union:
        return 1.0
    return len(set_t & set_t1) / len(union)


def feature_switch_rate(feature_sets_over_time: Sequence[Sequence[str]], n_drift_events: int) -> float:
    """Number of feature-set changes divided by the number of drift
    events, over a sequence of consecutively selected feature sets.
    """
    if n_drift_events == 0:
        return 0.0
    changes = sum(
        1 for prev, curr in zip(feature_sets_over_time[:-1], feature_sets_over_time[1:])
        if set(prev) != set(curr)
    )
    return changes / n_drift_events


def feature_importance_consistency(rank_t: Sequence[float], rank_t_minus_1: Sequence[float]) -> float:
    """Spearman rank correlation of feature-importance rankings between
    two consecutive windows (same features, same order expected).
    """
    if len(rank_t) < 2 or len(rank_t_minus_1) < 2:
        return float("nan")
    corr, _ = spearmanr(rank_t, rank_t_minus_1)
    return float(corr)


# ---------------------------------------------------------------------------
# Runtime measurement (used everywhere)
# ---------------------------------------------------------------------------

@contextmanager
def timer() -> Iterator[Dict[str, float]]:
    """Context manager for measuring wall-clock runtime.

    Usage:
        with timer() as t:
            do_something()
        print(t["seconds"])
    """
    result: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
