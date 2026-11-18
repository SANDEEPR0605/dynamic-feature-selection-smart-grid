# Dynamic Feature Selection for Non-Stationary Environments

A drift-triggered framework for smart grid load forecasting.

Static feature selection picks a fixed set of input variables once and never revisits it, which is a poor fit when the relationship between features and electricity load changes over time. This project builds a framework that detects concept drift and responds by updating both its active feature subset and its regression model. It compares that framework against a static baseline and a continuously updating online learner, all evaluated on the same held-out stream.

## Approach

- **Final framework:** XGBoost regressor, ADWIN drift detector and Random Forest feature-importance selection.
- **Compared in sequence:** three regression models, three drift detectors and three feature-selection methods, then validated end to end.
- **Online baseline:** a continuously updating online regressor built with [River](https://riverml.xyz).
- **Statistics:** paired Wilcoxon tests plus an autocorrelation-robust block-bootstrap test, because the evaluation stream is temporally autocorrelated.

## Repository layout

```
project/
├── notebooks/    One notebook per experimental phase
├── src/          Shared Python modules imported by every notebook
├── data/
│   ├── raw/        The original Kaggle CSV, unmodified (not tracked by Git)
│   └── processed/  Cleaned, time-ordered dataset (written by Phase 1)
└── results/
    ├── tables/     CSV outputs
    ├── figures/    Plots
    └── logs/       Per-window experiment logs
```

## Phases

| Phase | Notebook | Modules | Purpose |
|---|---|---|---|
| 1 | `01_data_preparation` | `data_utils.py` | Load, clean, order and split the data |
| 2 | `02_model_comparison` | `models.py`, `metrics.py` | Linear Regression vs Random Forest vs XGBoost |
| 3 | `03_window_sensitivity` | | Window-length sensitivity analysis |
| 4 | `04_drift_detection` | `drift_detectors.py` | ADWIN vs DDM vs Page-Hinkley |
| 5 | `05_feature_selection` | `feature_selection.py` | Sliding-window correlation vs RF importance vs permutation importance |
| 6 | `06_combination_validation` | `dynamic_framework.py` | End-to-end validation of method combinations |
| 7 | `07_dynamic_framework` | | Final drift-triggered framework |
| 8 | `08_online_learner` | `online_learner.py` | Online learning benchmark |
| 9 | `09_final_comparison` | | Static vs dynamic vs online, with statistical tests |

All nine phases are complete: the pipeline runs end to end from the raw CSV to the final statistically tested three-way comparison.

## Results by phase

**Phase 2: model comparison.** Random Forest and XGBoost perform almost identically (MAE about 1.42 kW, R² about 0.65), both clearly ahead of Linear Regression (MAE about 1.56 kW, R² about 0.60). The Random Forest is depth-capped (`max_depth=15`, `min_samples_leaf=3`), because an unbounded forest on about 40,000 rows serialises to several hundred MB with negligible accuracy gain.

**Phase 3: window length.** Feature-importance stability is model-free, so it is swept over the full 50,000-row dataset (up to 6,000-row windows), while predictive accuracy is evaluated on the test stream only. Stability was still rising at the largest size, so a diminishing-returns (elbow) rule was used: the smallest window capturing at least 80% of the achievable stability gain. This selects 4,000 rows, which still allows 12 adaptation checks across the stream.

**Phase 4: drift detection.** ADWIN, DDM and Page-Hinkley were run on the model's per-row error stream, against three injected drift points with known ground truth. A first ranking by detections and delay picked Page-Hinkley (8.3-sample delay) despite 62 false positives. Two fixes followed: Page-Hinkley's threshold was reset to River's documented default (50.0), which cut false positives to 29, and selection became two-tier, with a usability filter (false positives must not exceed the number of true drifts) applied before ranking. Result: ADWIN detected 3/3 drifts with 0 false positives and a mean delay of about 76 samples. DDM detected 1/3 with 2 false positives. Page-Hinkley stays excluded.

**Phase 5: feature selection.** All three methods perform almost identically on downstream MAE (3.059, 3.068 and 3.076 kW), so Random Forest importance is ranked first without being a confident win. Sliding-window correlation is roughly 1,000 times cheaper to compute for statistically indistinguishable accuracy. A bias check found no model-specific bias: RF-selected features scored 3.10 kW with XGBoost against 3.06 kW with Random Forest, within the 10% tolerance.

**Phase 6: combination validation.** Swapping XGBoost in for Random Forest improved MAE by 9.8% (past the 5% materiality threshold) and ran about 2.7 times faster, even though the two models were tied as static models. The final combination is **XGBoost + ADWIN + RF importance**.

**Phase 7: dynamic framework.** On the synthetic drift stream, drift-triggered adaptation improved MAE by 7.0% against static XGBoost (2.313 to 2.151 kW). Only 2 of the 3 injected drifts triggered a retrain here, against 3/3 in Phase 4, because ADWIN was watching XGBoost's error stream instead of Random Forest's.

**Phase 8: online learner.** River's Hoeffding Tree Regressor failed badly with default hyperparameters (R² = -1823), and the preliminary selection step caught it. Online Lasso (R² = 0.55) was selected.

**Phase 9: final comparison** (identical 10,000-row synthetic stream):

| System | MAE (kW) | vs static | Runtime (s) |
|---|---|---|---|
| Static baseline (Random Forest) | 2.317 | n/a | 0.24 |
| Dynamic framework (XGBoost + ADWIN + RF importance) | 2.151 | -7.2% | 6.53 |
| Online learner (Online Lasso) | 1.897 | -18.1% | 0.22 |

The ranking by point estimate is static < dynamic < online, the reverse of the ranking by complexity and cost.

### Statistical testing

A standard paired Wilcoxon signed-rank test assumes independent pairs, which the temporally autocorrelated stream violates, and it returned implausibly small p-values. An autocorrelation-robust block-bootstrap test (block size 200 rows, 5,000 resamples) was therefore also run:

| Comparison | Mean error diff. (kW) | Wilcoxon p | Block-bootstrap p |
|---|---|---|---|
| Static vs dynamic | 0.166 | 6.8e-39 | 0.142 |
| Dynamic vs online | 0.254 | 3.0e-17 | 0.089 |
| Static vs online | 0.420 | 7.8e-101 | 1.9e-9 |

Only the static-vs-online difference survives the block-bootstrap test. The dynamic framework's 7.2% MAE reduction is directionally consistent with the aim of the project, but its statistical evidence is much weaker than the Wilcoxon test suggested, and it should not be read as conclusively significant on this dataset.

## Getting started

1. Clone the repository and install the dependencies:
   ```bash
   git clone https://github.com/YOUR-USERNAME/dynamic-feature-selection-smart-grid.git
   cd dynamic-feature-selection-smart-grid
   python -m venv .venv
   .venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
2. Download the [Smart Grid Real-Time Load Monitoring Dataset](https://www.kaggle.com/datasets/ziya07/smart-grid-real-time-load-monitoring-dataset) from Kaggle (50,000 rows, 16 columns) and save it as `data/raw/smart_grid_dataset.csv`. The dataset is not redistributed here; please check its terms on Kaggle.
3. Launch Jupyter from the `notebooks/` folder and run the notebooks in order, 01 to 09:
   ```bash
   cd notebooks
   jupyter notebook
   ```

The notebooks also run on Google Colab, where they mount Google Drive.
