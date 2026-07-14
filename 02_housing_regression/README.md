<div align="center">

# 🏡 California Housing Regression

**A leakage-proof scikit-learn pipeline for median house values.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

</div>

---

Predicts median house values from the built-in California Housing dataset. The focus was a clean `Pipeline` that handles outliers and skew properly, rather than throwing a model at raw features.

## 🧠 How it works

1. **Load** — `fetch_california_housing` pulls the data directly (scikit-learn caches it on first run, so there's no local file to manage).
2. **Remove outliers** — rows with any feature outside 1.5× the IQR for that column are dropped.
3. **Transform the target** — `MedHouseVal` is log-transformed with `log1p` to tame its skew.
4. **Pipeline** — a `QuantileTransformer` (features → normal distribution) feeds a `HistGradientBoostingRegressor` (`learning_rate=0.05`, `max_depth=6`, `max_iter=800`, `min_samples_leaf=10`, `l2_regularization=0.1`).
5. **Evaluate** — fit on an 80/20 split; R² and RMSE computed on the log-scale test predictions.

## 📊 Results

From `results/results.txt`:

| Metric | Value |
| --- | --- |
| Train / test shape | (13473, 8) / (3369, 8) |
| Test R² | **0.8452** |
| Test RMSE (log scale) | 0.1330 |

`results/prediction_analysis.png` plots predicted vs. actual log house values against a perfect-prediction reference line.

## 🚀 Getting started

```bash
uv sync
uv run main.py
```

Writes `results/results.txt` and `results/prediction_analysis.png`.
