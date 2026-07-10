# California Housing Regression

A scikit-learn pipeline that predicts median house values from the built-in
California Housing dataset. The focus here was building a clean `Pipeline`
that handles outliers and skew properly instead of just throwing a model at
raw features.

## How it works

1. **Load the data** - `sklearn.datasets.fetch_california_housing` pulls the
   dataset directly (scikit-learn downloads and caches it on first run, so
   there's no local data file to manage).
2. **Remove outliers** - Any row where a feature falls outside 1.5x the IQR
   (interquartile range) for that column gets dropped.
3. **Transform the target** - The target (`MedHouseVal`) is log-transformed
   with `log1p` to tame its skew before fitting.
4. **Pipeline** - A single `sklearn.pipeline.Pipeline` chains a
   `QuantileTransformer` (mapping features to a normal distribution) into a
   `HistGradientBoostingRegressor` (`learning_rate=0.05`, `max_depth=6`,
   `max_iter=800`, `min_samples_leaf=10`, `l2_regularization=0.1`).
5. **Evaluate** - The pipeline is fit on an 80/20 train/test split, and R2 and
   RMSE are computed on the (log-scale) test predictions.

## Results

From `results/results.txt`:

- Train shape: (13473, 8)
- Test shape: (3369, 8)
- Test R2: 0.8452
- Test RMSE (log scale): 0.1330

`results/prediction_analysis.png` plots predicted vs. actual log house values
against a perfect-prediction reference line.

## Getting started

```bash
uv sync
uv run src/main.py
```

This writes `results/results.txt` and `results/prediction_analysis.png`.
