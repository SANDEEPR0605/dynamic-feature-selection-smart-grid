"""
src/feature_selection.py

Feature-selection methods for the "Dynamic Feature Selection for
Non-Stationary Environments" dissertation project.

Implements Chapter 3, Section 3.6 and Chapter 4, Section 4.7: sliding-
window correlation (model-free), Random Forest importance (embedded,
requires a fitted Random Forest), and permutation importance (wrapper-
style, requires a fitted model of any type). All three are explicitly
conditioned on the model selected in Phase 2 -- RF importance and
permutation importance require fitting a model on the window itself
(Chapter 3, Section 3.8: drift triggers "recompute feature importance ...
retrain model"), not the original Phase 2 model, since the whole point
is that relationships have shifted by the time a window is analysed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import data_utils as du
import models as mdl
import metrics as mt

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Method 1: Sliding-window correlation (Chapter 3, Section 3.6.3)
# Model-free -- reuses data_utils.windowed_correlation_ranking, computed
# only within the given window, consistent with the non-stationary
# premise of the dissertation.
# ---------------------------------------------------------------------------

def correlation_selection(
    window_df: pd.DataFrame, feature_cols: List[str], target_col: str, k: int,
) -> Tuple[List[str], pd.Series, float]:
    """Rank features by |windowed correlation| with the target; return the
    top-k feature names, the full ranking, and the runtime in seconds.
    """
    with mt.timer() as t:
        ranking = du.windowed_correlation_ranking(window_df, feature_cols, target_col)
        top_k = list(ranking.index[:k])
    return top_k, ranking, t["seconds"]


# ---------------------------------------------------------------------------
# Method 2: Random Forest importance (Chapter 3, Section 3.6.1)
# Embedded -- requires fitting a fresh Random Forest on the window.
# ---------------------------------------------------------------------------

def rf_importance_selection(
    window_df: pd.DataFrame, feature_cols: List[str], target_col: str, k: int,
) -> Tuple[List[str], pd.Series, float]:
    """Fit a fresh Random Forest on the window and rank features by its
    built-in feature_importances_. Returns the top-k feature names, the
    full ranking, and the runtime in seconds (fit + importance read-off).
    """
    X_win, y_win = window_df[feature_cols], window_df[target_col]
    with mt.timer() as t:
        rf = mdl.get_model("random_forest")
        rf.fit(X_win, y_win)
        ranking = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        top_k = list(ranking.index[:k])
    return top_k, ranking, t["seconds"]


# ---------------------------------------------------------------------------
# Method 3: Permutation importance (Chapter 3, Section 3.6.2)
# Wrapper-style -- requires a fitted model of any type. Restricted to
# drift-triggered windows only (never computed per-window generally),
# since it is markedly more expensive than the other two methods; the
# window is split into an internal train/validation portion so the
# permutation score reflects genuine held-out degradation rather than
# in-sample fit.
# ---------------------------------------------------------------------------

def permutation_importance_selection(
    window_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    k: int,
    base_model_name: str = "random_forest",
    val_fraction: float = 0.2,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[str], pd.Series, float]:
    """Fit `base_model_name` on the first (1 - val_fraction) of the window
    (in chronological order) and compute permutation importance on the
    remaining validation slice. Returns the top-k feature names, the full
    ranking, and the runtime in seconds (fit + permutation).
    """
    n = len(window_df)
    split = int(n * (1 - val_fraction))
    train_part, val_part = window_df.iloc[:split], window_df.iloc[split:]

    X_train, y_train = train_part[feature_cols], train_part[target_col]
    X_val, y_val = val_part[feature_cols], val_part[target_col]

    with mt.timer() as t:
        model = mdl.get_model(base_model_name)
        model.fit(X_train, y_train)
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats, random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        ranking = pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)
        top_k = list(ranking.index[:k])
    return top_k, ranking, t["seconds"]


# ---------------------------------------------------------------------------
# Downstream evaluation: retrain on the selected subset, evaluate forward
# ---------------------------------------------------------------------------

def evaluate_feature_subset(
    selected_features: List[str],
    window_df: pd.DataFrame,
    forward_df: pd.DataFrame,
    target_col: str,
    model_name: str,
) -> Dict[str, float]:
    """Fit `model_name` on `window_df` restricted to `selected_features`,
    then evaluate on `forward_df` (genuinely held-out future data, not
    seen during selection or fitting) -- this is what "downstream
    predictive performance" means in Chapter 3, Table 3.4.
    """
    X_train, y_train = window_df[selected_features], window_df[target_col]
    X_eval, y_eval = forward_df[selected_features], forward_df[target_col]

    model = mdl.get_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    return mt.compute_predictive_metrics(y_eval, y_pred, n_features=len(selected_features))
