"""
src/dynamic_framework.py

The integrated drift-triggered dynamic feature selection pipeline
(Chapter 3, Section 3.8, Figure 3.2; Chapter 4, Section 4.9).

This module is used in two places: Phase 6 (combination validation, this
notebook) runs it once per candidate (model, detector, feature-selection
method) combination to compare them end-to-end; Phase 7 runs it once more,
unchanged, with the final validated combination, to produce the proposed
framework's full results. Implementing the loop once here -- rather than
duplicating it -- means Phase 6's comparison and Phase 7's final run are
guaranteed to use identical simulation logic (Chapter 4, Section 4.2's
stated rationale for keeping shared logic in src/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import data_utils as du
import models as mdl
import metrics as mt
import drift_detectors as dd
import feature_selection as fsel

RANDOM_STATE = 42

FEATURE_SELECTION_FNS: Dict[str, Callable] = {
    "correlation": fsel.correlation_selection,
    "rf_importance": fsel.rf_importance_selection,
    "permutation_importance": fsel.permutation_importance_selection,
}


@dataclass
class PipelineResult:
    predictions: np.ndarray
    errors: np.ndarray
    y_true: np.ndarray
    drift_indices: List[int]
    feature_sets_at_drift: List[List[str]] = field(default_factory=list)
    rankings_at_drift: List[pd.Series] = field(default_factory=list)
    total_runtime_s: float = 0.0
    selection_runtime_s: float = 0.0
    n_retrains: int = 0
    final_feature_set: List[str] = field(default_factory=list)
    model_name: str = ""
    detector_name: str = ""
    feature_selection_method: str = ""
    window_size: int = 0


def run_dynamic_pipeline(
    stream_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
    detector_name: str,
    feature_selection_method: str,
    window_size: int,
    initial_model=None,
    top_k: int = 6,
    baseline_rows: int = 2000,
    min_window_for_reassessment: int = 30,
    predict_chunk_size: int = 250,
    random_state: int = RANDOM_STATE,
) -> PipelineResult:
    """Run the drift-triggered dynamic feature selection pipeline
    (Figure 3.2) over `stream_df`, row by row.

    At each row: predict with the current model and active feature set,
    compute the error, and feed it to the drift detector. On a drift
    signal, the most recent `window_size` rows (clipped near the start of
    the stream) are analysed with `feature_selection_method` to pick a
    new top-`top_k` feature subset, and a fresh `model_name` is retrained
    on that window restricted to the new subset -- consistent with
    Chapter 3, Section 3.8 ("drift -> analyse recent window -> recompute
    feature importance -> update feature subset -> retrain model").

    If `initial_model` is not given, a fresh model is fit on the first
    `window_size` rows of `stream_df` using the full feature set.

    Performance note: predictions are computed in vectorised batches of
    `predict_chunk_size` rows at a time (whatever model is currently
    active), rather than one row at a time -- a single-row .predict() call
    repeated thousands of times is dominated by per-call overhead rather
    than actual computation. Drift detection itself remains genuinely
    sequential (each row's error is fed to the detector in order, and a
    mid-chunk drift signal stops that chunk immediately), so this is a
    performance optimisation only: every row is still predicted using
    whichever model was actually active at that point in the stream, with
    no lookahead.
    """
    if feature_selection_method not in FEATURE_SELECTION_FNS:
        raise KeyError(f"Unknown feature-selection method '{feature_selection_method}'")
    selection_fn = FEATURE_SELECTION_FNS[feature_selection_method]

    n = len(stream_df)
    active_features = list(feature_cols)
    y_series = stream_df[target_col].to_numpy()

    with mt.timer() as total_t:
        if initial_model is not None:
            model = initial_model
        else:
            warm_df = stream_df.iloc[:window_size]
            model = mdl.get_model(model_name)
            model.fit(warm_df[active_features], warm_df[target_col])

        detector = dd.get_detector(detector_name)
        is_binary_detector = detector_name in dd.BINARY_DETECTORS

        # DDM's binarisation threshold is derived once, from the model's
        # error on the first `baseline_rows` rows only (assumed drift-free,
        # matching the baseline segment convention used in Phases 4-5).
        ddm_threshold = None
        if is_binary_detector:
            baseline_df = stream_df.iloc[:baseline_rows]
            baseline_errors = dd.compute_error_stream(model, baseline_df[active_features], baseline_df[target_col])
            ddm_threshold = baseline_errors.mean() + baseline_errors.std()

        predictions = np.empty(n)
        errors = np.empty(n)

        drift_indices: List[int] = []
        feature_sets_at_drift: List[List[str]] = []
        rankings_at_drift: List[pd.Series] = []
        selection_runtime_total = 0.0
        n_retrains = 0

        i = 0
        while i < n:
            chunk_end = min(i + predict_chunk_size, n)
            chunk_df = stream_df.iloc[i:chunk_end]
            chunk_preds = model.predict(chunk_df[active_features])

            drift_fired_at = None
            for offset, (pred, y_true) in enumerate(zip(chunk_preds, y_series[i:chunk_end])):
                row_idx = i + offset
                err = abs(y_true - pred)
                predictions[row_idx] = pred
                errors[row_idx] = err

                detector_input = float(err > ddm_threshold) if is_binary_detector else float(err)
                detector.update(detector_input)

                if detector.drift_detected:
                    drift_fired_at = row_idx
                    break

            if drift_fired_at is None:
                i = chunk_end
                continue

            # Drift fired mid-chunk: retrain immediately, then resume from
            # the next row with a fresh chunk under the new model.
            drift_indices.append(drift_fired_at)
            window_start = max(0, drift_fired_at - window_size)
            recent_window = stream_df.iloc[window_start:drift_fired_at]

            if len(recent_window) >= min_window_for_reassessment:
                top_k_features, ranking, sel_rt = selection_fn(recent_window, feature_cols, target_col, top_k)
                selection_runtime_total += sel_rt

                new_model = mdl.get_model(model_name)
                new_model.fit(recent_window[top_k_features], recent_window[target_col])

                model = new_model
                active_features = top_k_features
                n_retrains += 1
                feature_sets_at_drift.append(top_k_features)
                rankings_at_drift.append(ranking)

            i = drift_fired_at + 1

    return PipelineResult(
        predictions=predictions,
        errors=errors,
        y_true=y_series,
        drift_indices=drift_indices,
        feature_sets_at_drift=feature_sets_at_drift,
        rankings_at_drift=rankings_at_drift,
        total_runtime_s=total_t["seconds"],
        selection_runtime_s=selection_runtime_total,
        n_retrains=n_retrains,
        final_feature_set=active_features,
        model_name=model_name,
        detector_name=detector_name,
        feature_selection_method=feature_selection_method,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# Summarising a PipelineResult with the full Chapter 3, Table 3.5 metric suite
# ---------------------------------------------------------------------------

def summarize_result(result: PipelineResult, penalty_alpha: float = 1.5, adaptation_block: int = 100, adaptation_tolerance: float = 1.05) -> Dict[str, float]:
    """Compute the full evaluation metric suite (Chapter 3, Table 3.5) for
    a single PipelineResult.
    """
    acc = mt.compute_predictive_metrics(result.y_true, result.predictions, n_features=len(result.final_feature_set))
    penalised_mae = mt.penalised_mae(result.y_true, result.predictions, alpha=penalty_alpha)
    penalised_loss = mt.penalised_loss(result.y_true, result.predictions, alpha=penalty_alpha)

    # Interpretability metrics across the sequence of feature sets selected at drift events.
    stability_scores, consistency_scores = [], []
    for prev, curr in zip(result.feature_sets_at_drift[:-1], result.feature_sets_at_drift[1:]):
        stability_scores.append(mt.feature_stability_index(curr, prev))
    for prev_rank, curr_rank in zip(result.rankings_at_drift[:-1], result.rankings_at_drift[1:]):
        aligned_curr = curr_rank.reindex(prev_rank.index)
        consistency_scores.append(mt.feature_importance_consistency(aligned_curr.values, prev_rank.values))

    switch_rate = mt.feature_switch_rate(result.feature_sets_at_drift, len(result.drift_indices)) if result.drift_indices else 0.0

    # Adaptation time: mean, over all drift events, of how many rows after
    # the drift point it takes for the block-averaged error to return to
    # (or below) its pre-drift baseline level.
    adaptation_times = []
    for drift_idx in result.drift_indices:
        pre_start = max(0, drift_idx - adaptation_block)
        pre_drift_error = float(np.mean(result.errors[pre_start:drift_idx])) if drift_idx > pre_start else float("nan")
        if np.isnan(pre_drift_error):
            continue
        post_errors = result.errors[drift_idx:]
        block_means = [
            float(np.mean(post_errors[b:b + adaptation_block]))
            for b in range(0, len(post_errors), adaptation_block)
        ]
        at = mt.adaptation_time(block_means, 0, pre_drift_error, tolerance=adaptation_tolerance)
        adaptation_times.append(at * adaptation_block if at is not None else None)

    detected_adaptation_times = [a for a in adaptation_times if a is not None]

    return {
        **acc,
        "Penalised_MAE": penalised_mae,
        "Penalised_Loss": penalised_loss,
        "total_runtime_s": result.total_runtime_s,
        "selection_runtime_s": result.selection_runtime_s,
        "n_drift_events": len(result.drift_indices),
        "n_retrains": result.n_retrains,
        "mean_feature_stability_index": float(np.mean(stability_scores)) if stability_scores else float("nan"),
        "mean_importance_consistency": float(np.nanmean(consistency_scores)) if consistency_scores else float("nan"),
        "feature_switch_rate": switch_rate,
        "mean_adaptation_time_rows": float(np.mean(detected_adaptation_times)) if detected_adaptation_times else float("nan"),
        "n_adaptation_events_recovered": len(detected_adaptation_times),
    }
