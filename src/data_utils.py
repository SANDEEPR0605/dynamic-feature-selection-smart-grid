"""
src/data_utils.py

Phase 1 data preparation utilities for the "Dynamic Feature Selection for
Non-Stationary Environments" dissertation project (Smart Grid Real-Time Load
Monitoring dataset, Kaggle).

Implements the steps set out in Chapter 3, Section 3.2 (Table 3.1) and
Chapter 4, Section 4.3:
    1. Load the raw dataset
    2. Verify missing values and data types
    3. Preserve chronological ordering
    4. Identify the target variable (Predicted Load (kW))
    5. Separate predictor variables
    6. Remove pure identifier/index variables (timestamp retained for
       ordering only, never as a model input)
    7. Report global correlation and VIF as descriptive diagnostics only
       (never used to eliminate features)
    8. Define the full candidate feature set
    9. Create the temporally ordered training/evaluation split

Also provides a generic sliding-window generator, shared by every later
phase (window-length sensitivity, drift detection, feature selection, the
integrated dynamic framework) so that "what counts as a window" is defined
in exactly one place.

All functions operate on data that is assumed to already be in, or is
explicitly sorted into, chronological order. No function in this module
shuffles rows.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Configuration -- matches the actual columns in smart_grid_dataset.csv
# ---------------------------------------------------------------------------

TIMESTAMP_COL = "Timestamp"
TARGET_COL = "Predicted Load (kW)"

# Excluded as target leakage (Chapter 4, Section 4.3). Confirmed empirically
# on the real dataset: corr(Current (A), target) = 0.996,
# corr(Power Consumption (kW), target) = 0.998 -- both are effectively the
# target under a different name (Power Consumption (kW) approx
# Voltage x Current x Power Factor), so a model given either would be
# looking up the answer rather than forecasting it.
LEAKAGE_COLUMNS: List[str] = ["Power Consumption (kW)", "Current (A)"]

# Identifier / ordering-only column -- never passed to a model as a feature
# (Chapter 3, Table 3.1, Step 6).
IDENTIFIER_COLUMNS: List[str] = [TIMESTAMP_COL]

# The twelve candidate features once the target, the timestamp, and the two
# leakage columns are removed (Chapter 4, Section 4.3). Spans the
# electrical, renewable-energy, grid-supply, environmental, and economic
# categories set out in Chapter 1, plus grid-condition indicators
# (Overload Condition, Transformer Fault, Voltage Fluctuation, Power
# Factor) present in the real dataset but not anticipated in the original
# project outline.
CANDIDATE_FEATURES: List[str] = [
    "Voltage (V)",
    "Reactive Power (kVAR)",
    "Power Factor",
    "Solar Power (kW)",
    "Wind Power (kW)",
    "Grid Supply (kW)",
    "Voltage Fluctuation (%)",
    "Overload Condition",
    "Transformer Fault",
    "Temperature (\u00b0C)",
    "Humidity (%)",
    "Electricity Price (USD/kWh)",
]


# ---------------------------------------------------------------------------
# Steps 1-2: load and inspect
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV exactly as downloaded from Kaggle (Step 1).

    No cleaning or reordering happens here -- this is a thin read so that
    inspect_data() can report the *actual* state of the file before any
    changes are made.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. Place the Kaggle CSV at "
            f"data/raw/ and pass its path to load_raw_data()."
        )
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values, dtypes, and cardinality per column (Step 2).

    Returns a summary DataFrame rather than printing, so the notebook can
    both display it and save it to results/tables/ for Chapter 4/5.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().sum() / len(df) * 100).round(3),
        "n_unique": df.nunique(),
    })
    return summary


# ---------------------------------------------------------------------------
# Step 3: chronological ordering
# ---------------------------------------------------------------------------

def enforce_chronological_order(
    df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL, parse_dates: bool = True
) -> pd.DataFrame:
    """Sort the dataset strictly by timestamp (Step 3).

    This is the single most important guarantee in the whole pipeline:
    every later phase (windowing, drift detection, temporal splitting)
    assumes row order == time order. Warns (rather than silently guessing)
    if duplicate timestamps make that ordering ambiguous.
    """
    out = df.copy()
    if parse_dates:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col])

    n_dupes = int(out[timestamp_col].duplicated().sum())
    if n_dupes > 0:
        warnings.warn(
            f"{n_dupes} duplicate timestamp(s) found. Rows will be ordered "
            f"by timestamp with original row order preserved as a tiebreak "
            f"(stable sort); consider investigating whether these are "
            f"genuine duplicate readings."
        )

    out = out.sort_values(timestamp_col, kind="mergesort")  # stable sort
    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Steps 4-6: target, predictors, identifiers, leakage
# ---------------------------------------------------------------------------

def split_target_and_predictors(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate predictors (X) from the target (y) (Steps 4-5).

    `drop_cols` should contain identifier columns and any leakage columns
    (see LEAKAGE_COLUMNS / IDENTIFIER_COLUMNS above) -- this function does
    not decide what counts as leakage, it only removes what it is told to.
    """
    drop_cols = drop_cols or []
    missing = [c for c in [target_col, *drop_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in dataframe: {missing}")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col, *drop_cols])
    return X, y


# ---------------------------------------------------------------------------
# Step 7: descriptive diagnostics only (correlation / VIF) -- Chapter 3,
# Section 3.2: never used to eliminate features.
# ---------------------------------------------------------------------------

def compute_correlation_with_target(
    df: pd.DataFrame, feature_cols: List[str], target_col: str = TARGET_COL
) -> pd.Series:
    """Global Pearson correlation of each candidate feature with the target.

    Descriptive only (Chapter 3, Section 3.2) -- also doubles as the
    empirical check used to justify excluding the leakage columns in
    Section 4.3 (a candidate feature with |corr| > ~0.98 against the
    target is treated as a leakage suspect and inspected manually, not
    auto-dropped).
    """
    corrs = df[feature_cols + [target_col]].corr(method="pearson")[target_col]
    return corrs.drop(target_col).sort_values(ascending=False)


def compute_correlation_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Global pairwise Pearson correlation across the candidate features.

    Descriptive only -- reported in Chapter 5's dataset-characterisation
    table, never used here to drop a feature.
    """
    return df[feature_cols].corr(method="pearson")


def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Global Variance Inflation Factor per candidate feature.

    Descriptive only. A high VIF here does not trigger removal: the whole
    point of Section 3.2 is that global collinearity does not imply local
    (within-window) collinearity in a non-stationary stream, so any action
    on this diagnostic is deferred to the windowed feature-selection
    methods in Phase 5, not taken here.
    """
    X = df[feature_cols].dropna().astype(float)
    X = X.assign(_intercept=1.0)  # VIF is conventionally computed with an intercept term
    vifs = []
    for i, col in enumerate(feature_cols):
        vif_val = variance_inflation_factor(X.values, i)
        vifs.append({"feature": col, "VIF": vif_val})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 8: candidate feature set
# ---------------------------------------------------------------------------

def get_candidate_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """Return the full candidate feature set (Step 8).

    Per Chapter 3 (Section 3.2) and Chapter 4 (Section 4.3): this is *all*
    remaining predictors after removing the target, identifier columns, and
    the two target-leakage columns identified during implementation
    (Power Consumption (kW), Current (A)) -- no relevance-based pruning
    happens here. Relevance is determined downstream by the
    feature-selection methods (Phase 5), not by an upfront filter.
    """
    exclude_cols = exclude_cols or []
    return [c for c in df.columns if c not in [target_col, *exclude_cols]]


# ---------------------------------------------------------------------------
# Step 9: temporal (non-random) split
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: the earliest (1 - test_size) fraction of rows
    become the training/development stream, the most recent test_size
    fraction become the held-out evaluation stream.

    Deliberately not sklearn's train_test_split -- that shuffles by
    default, which would destroy the temporal structure this entire
    dissertation depends on.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Shared windowing utility (used by Phases 3-7)
# ---------------------------------------------------------------------------

def generate_windows(
    df: pd.DataFrame, window_size: int, step: Optional[int] = None
) -> Iterator[Tuple[int, int, pd.DataFrame]]:
    """Yield consecutive (start_idx, end_idx, window_df) triples.

    Non-overlapping by default (step == window_size). A smaller step would
    produce overlapping windows -- exposed as a parameter since later
    phases (e.g. the window-length sensitivity analysis, Section 3.4) may
    want to experiment with overlap, but the default matches the simple
    fixed-size batching described in Chapter 1 / Chapter 3.

    `df` must already be in chronological order (see
    enforce_chronological_order) -- this function does not check or sort.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    step = step or window_size

    n = len(df)
    for start in range(0, n, step):
        end = min(start + window_size, n)
        yield start, end, df.iloc[start:end]
        if end == n:
            break


# ---------------------------------------------------------------------------
# Windowed correlation ranking -- a lightweight, model-free feature
# relevance signal used here (Phase 3, Section 3.4) as a proxy for feature
# stability, and reused in Phase 5 as the sliding-window correlation
# feature-selection method (Chapter 3, Section 3.6.3). Unlike Step 7's
# global correlation, this is computed within a single window, since the
# whole premise of the dissertation is that feature relevance is local
# and time-varying, not global and fixed.
# ---------------------------------------------------------------------------

def windowed_correlation_ranking(
    window_df: pd.DataFrame, feature_cols: List[str], target_col: str = TARGET_COL
) -> pd.Series:
    """Rank candidate features by |Pearson correlation| with the target,
    computed only within the given window.

    Returns a Series indexed by feature name, sorted descending by
    absolute correlation (most relevant first). Features with zero
    variance within the window (e.g. a constant flag column in a short
    window) yield NaN correlation and are sorted last.
    """
    corrs = window_df[feature_cols + [target_col]].corr(method="pearson")[target_col]
    corrs = corrs.drop(target_col)
    return corrs.reindex(corrs.abs().sort_values(ascending=False, na_position="last").index)


# ---------------------------------------------------------------------------
# Save processed output
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Persist the cleaned, time-ordered dataset to data/processed/."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
