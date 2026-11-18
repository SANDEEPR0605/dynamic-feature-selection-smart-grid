# Dynamic Feature Selection for Non-Stationary Environments

A drift-triggered framework for smart grid load forecasting.

Static feature selection picks a fixed set of input variables once and never revisits it, which is a poor fit when the relationship between features and electricity load changes over time. This project builds a framework that detects concept drift and responds by updating both its active feature subset and its regression model. It compares that framework against a static baseline and a continuously updating online learner, all evaluated on the same held-out stream.

## Approach

- **Final framework:** XGBoost regressor, ADWIN drift detector and Random Forest feature-importance selection.
- **Compared in sequence:** three regression models, three drift detectors and three feature-selection methods, then validated end to end.
- **Online baseline:** a continuously updating online regressor built with [River](https://riverml.xyz).
- **Statistics:** paired tests plus an autocorrelation-robust block-bootstrap test, because the evaluation stream is temporally autocorrelated.

## Headline result

The dynamic framework reduced mean absolute error by 7.2% (point estimate) against a static Random Forest baseline. The more conservative block-bootstrap test found that this difference is not statistically distinguishable from zero on this dataset.

## Repository layout

```
notebooks/   Nine notebooks, run in order (one per project phase)
src/         Helper modules imported by the notebooks
data/raw/    Place the downloaded dataset here
results/     Tables, figures, logs and models written by the notebooks
```

| Notebook | Phase |
|---|---|
| `01_data_preparation` | Load, clean, order and split the data |
| `02_model_comparison` | Compare Linear Regression, Random Forest and XGBoost |
| `03_window_sensitivity` | Training-window size sensitivity |
| `04_drift_detection` | Compare drift detectors on synthetic and real streams |
| `05_feature_selection` | Compare feature-selection methods |
| `06_combination_validation` | End-to-end validation of method combinations |
| `07_dynamic_framework` | The dynamic feature selection framework |
| `08_online_learner` | Online learning baseline |
| `09_final_comparison` | Static vs dynamic vs online, with statistical tests |

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

Dynamic Feature Selection for Non-Stationary Environments — Project Code

project/
├── notebooks/     One notebook per experimental phase (Chapter 3, Table 3.6)
├── src/           Shared, reusable Python modules imported by every notebook
├── data/
│   ├── raw/       The original Kaggle CSV, unmodified
│   └── processed/ Cleaned, time-ordered dataset (written by Phase 1)
└── results/
    ├── tables/    CSV outputs consumed by Chapter 5's tables
    ├── figures/   Plots consumed by Chapter 5
    └── logs/      Per-window experiment logs
Status
Phase 1 — Data Preparation (01_data_preparation.ipynb, src/data_utils.py)
Phase 2 — Model Comparison (02_model_comparison.ipynb, src/models.py, src/metrics.py)
Phase 3 — Window-Length Sensitivity Analysis (03_window_sensitivity.ipynb)
Phase 4 — Drift Detector Comparison (04_drift_detection.ipynb, src/drift_detectors.py)
Phase 5 — Feature-Selection Comparison (05_feature_selection.ipynb, src/feature_selection.py)
Phase 6 — Combination Validation (06_combination_validation.ipynb, src/dynamic_framework.py)
Phase 7 — Dynamic Framework Integration (07_dynamic_framework.ipynb)
Phase 8 — Online Learning Benchmark (08_online_learner.ipynb, src/online_learner.py)
Phase 9 — Final Comparison (09_final_comparison.ipynb)
All 9 phases complete. The full pipeline, from raw CSV to final statistically-tested three-way comparison, runs end-to-end.
Phase 4 — Drift Detector Comparison
Phase 5 — Feature-Selection Comparison
Phase 6 — Combination Validation
Phase 7 — Dynamic Framework Integration
Phase 8 — Online Learning Benchmark
Phase 9 — Final Comparison
Each subsequent phase will add one notebook to notebooks/ and one module to src/ — just drop the new files into the matching folders in your Drive and continue running from where you left off.
Phase 2 result (for reference)
On this dataset, Random Forest and XGBoost perform almost identically (MAE ≈ 1.42 kW, R² ≈ 0.65), both clearly ahead of Linear Regression (MAE ≈ 1.56 kW, R² ≈ 0.60). Random Forest is marginally best and is recorded as best_model in results/tables/phase2_artifacts.json; XGBoost is second_best_model (used in Phase 6's combination validation) and also serves as the non-Random-Forest model for the RF-importance bias check in Phase 5 (Chapter 3, Section 3.6.1).
Note: the Random Forest is deliberately depth-capped (max_depth=15, min_samples_leaf=3) — an unbounded forest on ~40,000 rows serialises to several hundred MB with negligible accuracy gain, which is impractical to version across phases.
Phase 3 result (for reference)
The first version of this analysis (window sizes up to 1,500 rows, tested only on the 10,000-row test stream) found stability still rising with no plateau — an inconclusive result. This was fixed by decoupling the two sweeps: feature-importance stability is model-free, so it now runs over the full 50,000-row dataset across a much wider range (up to 6,000 rows), while predictive accuracy still runs on the test stream only.
Even at 6,000 rows, consistency was still technically rising — but picking "closest to the maximum" on a monotonically increasing curve just selects the largest size tested regardless of whether it's sensible (6,000 rows gives only 8 adaptation-check windows across the whole dataset, undermining the point of a dynamic framework). Instead, a diminishing-returns (elbow) rule was used: the smallest window size capturing ≥80% of the achievable stability improvement. This selects 4,000 rows — capturing most of the available stability gain while still allowing 12 adaptation-check windows across the full stream, a much more defensible trade-off than either extreme.
Phase 4 result (for reference)
ADWIN, DDM, and Page-Hinkley were each run on the same model's per-row error stream, evaluated against three synthetically injected drift points (known ground truth) and separately on the real, unmodified test stream.
A first pass ranked purely by "most detected, then lowest delay" and picked Page-Hinkley — which had the lowest delay (8.3 samples) but also 62 false positives against only 3 true events, the same kind of technically-satisfied-but-indefensible result as Phase 3. Two fixes were applied: Page-Hinkley's threshold was corrected from an unjustified custom value (30.0) to River's own library default (50.0) — cutting false positives to 29 — and the selection rule was changed to a two-tier design: a usability filter (false positives must not exceed the number of true drift points) applied before ranking by detection performance.
Result: ADWIN — 3/3 injected drifts detected, 0 false positives, mean detection delay ≈76 samples. Page-Hinkley is still excluded even at 29 false positives (Chapter 5 can report this as a finding, not just a footnote — it's a real illustration of the sensitivity/false-positive trade-off in drift detection). DDM detected only 1/3 with 2 false positives, ranking behind ADWIN among the usable detectors.
Phase 5 result (for reference)
Sliding-window correlation, Random Forest importance, and permutation importance were each evaluated at the three drift-triggered points where ADWIN actually fired (Phase 4's real alarm indices, not the hidden ground-truth injection points) — since ADWIN found zero drift on the real, unmodified stream, this comparison necessarily runs on the same synthetic stream used to validate detectors in Phase 4. That's stated plainly in the notebook rather than switched to quietly.
All three methods perform almost identically on downstream MAE (3.059 vs 3.068 vs 3.076 kW) — a negligible margin given only 3 data points. Random Forest importance is technically ranked first, but the notebook is explicit this isn't a confident win, just the best of three closely matched options. Sliding-window correlation is ~1,000x cheaper to compute (no model fitting required) for statistically indistinguishable accuracy — worth weighing against the raw ranking in Chapter 6, especially since the framework recomputes feature importance on every detected drift.
The RF-importance bias check (Section 3.6.1) found no evidence of model-specific bias: RF-importance-selected features scored 3.10 kW MAE with XGBoost vs 3.06 kW with Random Forest — within the 10% tolerance used to flag bias. The selected features appear genuinely transferable rather than an artefact of pairing RF importance with the RF model.
Phase 6 result (for reference) — this one matters
This is the most consequential result so far. src/dynamic_framework.py (the same drift-triggered simulation loop Phase 7 will use for the final framework) was run end-to-end for four combinations: the sequential pick (Random Forest + ADWIN + RF importance) and three alternatives, each swapping in the second-best choice for one component.
The sequential design was revised. Swapping in XGBoost (instead of Random Forest) improved MAE by 9.8% — well past the pre-specified 5% materiality threshold — and also ran ~2.7x faster (6.5s vs 17.4s). This matters because Phase 2 found Random Forest and XGBoost statistically tied in the static setting (MAE 1.418 vs 1.419). It's only once both are tested under actual drift-triggered retraining that XGBoost pulls ahead — a genuine justification for why the combination-validation step exists at all, not just a formality. Final combination for Phase 7: XGBoost + ADWIN + RF importance.
Also worth noting: mean_adaptation_time_rows is NaN for most combinations — no drift event recovered to its pre-drift error level within the stream. This is expected given the injection design (a sustained multiplicative target shift, not a temporary blip, so the post-drift regime has a genuinely different error scale even for a well-adapted model) and is reported as-is rather than hidden.
Phase 7 result (for reference)
The validated combination (XGBoost + ADWIN + RF importance) was run end-to-end on the synthetic stream, producing a full per-row log (10,000 rows: prediction, error, drift flag, active feature set) for Chapter 5. As a same-stream sanity check ahead of the official Phase 9 comparison, it was compared against static (no-adaptation) XGBoost: 7.0% MAE improvement (2.313 → 2.151 kW) from drift-triggered adaptation alone.
One honest divergence worth a sentence in Chapter 6: only 2 of the 3 true injected drift points triggered a retrain here, vs 3/3 in Phase 4's standalone detector test. The difference is the error stream — Phase 4 monitored Random Forest's errors, this run monitors XGBoost's, and ADWIN's sensitivity is relative to the specific signal it's watching. Model choice affects detector reliability, not just prediction accuracy.
Phase 8 result (for reference) — two findings worth flagging
1. The preliminary-selection step caught a genuinely broken model. River's Hoeffding Tree Regressor was catastrophic on this data with default hyperparameters: R² = -1823 (RMSE 128 kW, against a target that's mostly 0-15 kW). Online Lasso scored R² = 0.55. This isn't a bug in the code — it's exactly the scenario Chapter 4's "select empirically" design exists to catch, and it worked as intended. Worth a sentence in Chapter 4/6: the preliminary comparison step had real, not just theoretical, value here.
2. The online learner beat both the static baseline AND the dynamic framework. Online Lasso scored MAE 1.897 kW on the synthetic stream — better than Phase 7's dynamic framework (2.151) and the static baseline (2.313). This is a genuinely important result for Chapter 6: a simple linear model that updates every single row outperformed a much more complex drift-triggered batch-retraining scheme built around XGBoost. The error-over-time figure shows why — brief spikes right at each injected drift point, followed by fast recovery, consistent with continuous updating reacting almost immediately versus the batch approach's detect-then-retrain delay. This doesn't mean the dynamic framework has no value (interpretability, computational cost per prediction, and behaviour under different drift patterns are all still open questions for Chapter 6) — but it's an honest, unflattering-to-the- main-thesis result that Phase 9's official comparison will need to confront directly, not explain away.
Phase 9 result (for reference) — the final answer
All three systems compared on the identical synthetic stream, full metric suite, with paired Wilcoxon signed-rank tests (not just raw MAE differences) to ground the comparison statistically:
System
MAE (kW)
vs static
Runtime (s)
Static Baseline (Random Forest)
2.317
—
0.24
Dynamic Framework (XGBoost+ADWIN+RF Importance)
2.151
-7.2%, p ≈ 6.8e-39
6.53
Online Learner (Online Lasso)
1.897
-18.1%, p ≈ 7.8e-101
0.22

The result is genuinely two-sided, and both halves are statistically significant:
Dynamic beats static — supporting the dissertation's core hypothesis that drift-triggered adaptation helps (p ≈ 6.8e-39).
Online beats dynamic — a simple, continuously-updating linear model beats the more complex drift-triggered XGBoost framework (p ≈ 3.0e-17), while also running ~30x faster.
This is reported as-is in the notebook, not softened. It's a legitimate, defensible dissertation finding — "static < dynamic < online" on this dataset, with the added twist that the accuracy ranking is also the exact inverse of the complexity/cost ranking. That tension (is the dynamic framework's added complexity worth 7% over static, when a far simpler and cheaper method gets 18%?) is exactly what Chapter 6 needs to grapple with — interpretability, robustness to different drift types (not just the sustained-shift pattern tested here), and computational cost per prediction are the legitimate remaining angles, not a reason to explain the online-learner result away.

