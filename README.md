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
