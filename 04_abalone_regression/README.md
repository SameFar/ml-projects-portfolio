<div align="center">

# 🐚 Abalone Age Regression

**Predicting abalone age from physical measurements, three models compared.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)

</div>

---

Estimates an abalone's age from physical measurements instead of the usual method — cutting the shell, staining it, and counting rings under a microscope. Age is `rings + 1.5`, over the [Kaggle abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset).

## 🧠 How it works

1. **Load** — `data/abalone.data` is loaded and deduplicated. `Age = Rings + 1.5` is the target; rows with zero/negative `Height` (biologically impossible) are dropped.
2. **Feature engineering**
   - `Sex` one-hot encoded, dropping `Sex_I` to avoid collinearity.
   - `Volume` = Length × Diameter × Height
   - `Density` = Whole weight / Volume
   - `Meat_ratio` = Shucked weight / Whole weight
   - `Shell_ratio` = Shell weight / Whole weight
   - Rows that produce inf/NaN from those ratios are dropped.
3. **Target transform** — `Age` is log-transformed (`log1p`), converted back with `expm1` when scoring.
4. **Split & clean** — 75/25 split, then an `IsolationForest` (contamination=0.02) removes outliers from the training set only, so nothing leaks from the test set.
5. **Models** — `XGBRegressor`, `RandomForestRegressor`, and an `SVR` (in its own `StandardScaler` pipeline, since SVR is scale-sensitive).

## 📊 Results

From `results/results.txt` (MAE/MSE in years, after transforming back out of log space):

| Model | R² | MAE | MSE |
| --- | --- | --- | --- |
| XGBoost Regressor | 0.586 | 1.47 | 4.57 |
| Random Forest Regressor | 0.588 | 1.46 | 4.55 |
| Support Vector Regressor | 0.587 | 1.46 | 4.56 |

All three land around 1.46–1.47 years of average error, which suggests the ceiling here is set by how much these physical measurements can tell you about age — not by which model you pick.

## 🚀 Getting started

```bash
uv sync
uv run main.py
```

Prints the evaluation report and writes it to `results/results.txt`.
