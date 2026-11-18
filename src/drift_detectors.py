"""
src/drift_detectors.py

Concept drift detectors for the "Dynamic Feature Selection for
Non-Stationary Environments" dissertation project.

Implements Chapter 3, Section 3.5 and Chapter 4, Section 4.6: wrappers
around River's ADWIN, DDM, and Page-Hinkley behind a common interface,
plus the synthetic drift injection procedure (Section 3.5.1) used to
give ground-truth drift points for evaluating detection delay and
false-positive rate independently of downstream predictive accuracy.

Granularity note (a design decision not fully spelled out in Chapter 3):
these detectors monitor the model's prediction error PER ROW, not per
window. ADWIN and Page-Hinkley need dozens of observations before their
statistics stabilise (Page-Hinkley's default min_instances=30), and DDM
is even more data-hungry -- far more than the ~2-3 windows the 10,000-row
test stream would yield at the Phase 3 window length (4,000 rows). The
window length chosen in Phase 3 is used downstream (Phase 5 onward) to
define the "recent window" pulled for feature reassessment once a
detector fires -- a separate concern from the detector's own update
cadence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from river import drift

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Detector registry -- common interface over River's detectors.
# ADWIN and Page-Hinkley accept a continuous value directly (the absolute
# error). DDM is inherently binary (Section 2.6.2 / 3.5): it expects a
# 0/1 "was this an error" stream, so a threshold is used to binarise the
# continuous error signal for DDM only. The threshold is derived from the
# baseline (assumed drift-free) segment of the stream, documented at the
# call site rather than hard-coded here.
# ---------------------------------------------------------------------------

def make_adwin(delta: float = 0.002) -> drift.ADWIN:
    return drift.ADWIN(delta=delta)


def make_page_hinkley(min_instances: int = 30, delta: float = 0.005, threshold: float = 50.0) -> drift.PageHinkley:
    # threshold=50.0 is River's own library default, used deliberately rather
    # than a custom value, so the cross-detector comparison in Phase 4 starts
    # from each detector's own well-established default configuration.
    return drift.PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold)


def make_ddm(warm_start: int = 30, warning_threshold: float = 2.0, drift_threshold: float = 3.0) -> drift.binary.DDM:
    return drift.binary.DDM(warm_start=warm_start, warning_threshold=warning_threshold, drift_threshold=drift_threshold)


DETECTOR_REGISTRY = {
    "adwin": make_adwin,
    "ddm": make_ddm,
    "page_hinkley": make_page_hinkley,
}

DETECTOR_DISPLAY_NAMES = {
    "adwin": "ADWIN",
    "ddm": "DDM",
    "page_hinkley": "Page-Hinkley",
}

# Which detectors need a binarised (0/1) input rather than the continuous
# error value directly.
BINARY_DETECTORS = {"ddm"}


def get_detector(name: str):
    if name not in DETECTOR_REGISTRY:
        raise KeyError(f"Unknown detector '{name}'. Options: {list(DETECTOR_REGISTRY)}")
    return DETECTOR_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Error stream computation
# ---------------------------------------------------------------------------

def compute_error_stream(model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Per-row absolute error of `model` on (X, y), in original row order."""
    y_pred = model.predict(X)
    return np.abs(np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float))


def binarize_error_stream(error_stream: np.ndarray, threshold: float) -> np.ndarray:
    """1 = 'error' (above threshold), 0 = 'correct' (at or below threshold).

    Used only for DDM (Section 3.5), which is inherently error-rate-based.
    """
    return (error_stream > threshold).astype(int)


# ---------------------------------------------------------------------------
# Running a single detector over a stream
# ---------------------------------------------------------------------------

def run_detector_on_stream(name: str, values: Sequence[float]) -> List[int]:
    """Feed `values` to a fresh instance of the named detector, one at a
    time, and return the list of row indices (0-based) at which drift was
    signalled.
    """
    detector = get_detector(name)
    drift_points = []
    for i, v in enumerate(values):
        detector.update(v)
        if detector.drift_detected:
            drift_points.append(i)
    return drift_points


# ---------------------------------------------------------------------------
# Synthetic drift injection (Chapter 3, Section 3.5.1)
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    df: pd.DataFrame
    true_drift_indices: List[int]
    segment_bounds: List[Tuple[int, int]] = field(default_factory=list)


def inject_synthetic_drift(
    stream_df: pd.DataFrame,
    target_col: str,
    injection_points: Sequence[int],
    scale_factors: Sequence[float],
    random_state: int = RANDOM_STATE,
) -> InjectionResult:
    """Inject known, labelled drift points into a copy of `stream_df` by
    applying a sustained multiplicative shift to the target variable from
    each injection point onward, up to the next injection point.

    A multiplicative target shift is used (rather than perturbing the
    features) because it reliably breaks the *already-fitted* model's
    learned input-output relationship -- causing predictable, sustained
    error increases -- while leaving the feature values themselves
    realistic. This is a synthetic stand-in for a genuine regime change
    (e.g. a sustained shift in typical load levels), not a claim that the
    real dataset contains this exact pattern.

    `injection_points` are row indices (0-based, sorted ascending) where
    a new segment (and a new scale factor) begins. `scale_factors` must
    have one entry per segment, i.e. len(injection_points) + 1 (the first
    segment, before the first injection point, is left unscaled: use 1.0
    as its factor if you want the segment list to be explicit, or omit it
    -- see below).

    Returns the perturbed dataframe, the list of true drift indices
    (equal to `injection_points`), and the segment boundaries used.
    """
    if len(scale_factors) != len(injection_points) + 1:
        raise ValueError(
            f"Expected {len(injection_points) + 1} scale factors "
            f"(one per segment: baseline + one per injection point), got {len(scale_factors)}"
        )

    df = stream_df.reset_index(drop=True).copy()
    n = len(df)
    bounds = [0, *injection_points, n]
    segment_bounds = list(zip(bounds[:-1], bounds[1:]))

    for (start, end), factor in zip(segment_bounds, scale_factors):
        df.loc[start:end - 1, target_col] = df.loc[start:end - 1, target_col] * factor

    return InjectionResult(df=df, true_drift_indices=list(injection_points), segment_bounds=segment_bounds)


# ---------------------------------------------------------------------------
# Evaluating detection delay and false-positive rate against ground truth
# ---------------------------------------------------------------------------

def evaluate_against_ground_truth(
    detected_indices: Sequence[int],
    true_drift_indices: Sequence[int],
    tolerance: int = 500,
) -> Dict[str, object]:
    """Compare a detector's alarm indices against known true drift points.

    For each true drift point, the detection delay is the distance to the
    nearest alarm that falls within `tolerance` samples AFTER it (alarms
    before a true drift point cannot be attributed to it). A true drift
    point with no such alarm is recorded as undetected (delay = None).

    Any alarm that is not within `tolerance` samples after any true drift
    point is counted as a false positive.
    """
    detected_indices = sorted(detected_indices)
    delays: List[Optional[int]] = []
    claimed = set()

    for true_idx in true_drift_indices:
        candidates = [d for d in detected_indices if true_idx <= d <= true_idx + tolerance]
        if candidates:
            best = min(candidates)
            delays.append(best - true_idx)
            claimed.add(best)
        else:
            delays.append(None)

    false_positives = [d for d in detected_indices if d not in claimed]

    n_detected = sum(1 for d in delays if d is not None)
    mean_delay = float(np.mean([d for d in delays if d is not None])) if n_detected else float("nan")

    return {
        "delays": delays,
        "n_true_drifts": len(true_drift_indices),
        "n_detected": n_detected,
        "mean_detection_delay": mean_delay,
        "false_positive_indices": false_positives,
        "n_false_positives": len(false_positives),
    }
