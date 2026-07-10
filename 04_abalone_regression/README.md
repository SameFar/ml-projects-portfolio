# Abalone Age Regression

Predicts an abalone's age from physical measurements instead of the usual
method — cutting the shell, staining it, and counting rings under a
microscope. Age is just rings + 1.5, so this is a regression problem over
the [Kaggle abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset).

## How it works

1. **Load the data** - `data/abalone.data` is loaded and deduplicated.
   `Age = Rings + 1.5` becomes the target, and rows with zero or negative
   `Height` (biologically impossible) are dropped.
2. **Feature engineering**
   - `Sex` is one-hot encoded, dropping `Sex_I` to avoid collinearity.
   - `Volume` = Length x Diameter x Height
   - `Density` = Whole weight / Volume
   - `Meat_ratio` = Shucked weight / Whole weight
   - `Shell_ratio` = Shell weight / Whole weight
   - Any row that ends up with an inf or NaN from those ratios is dropped.
3. **Target transform** - `Age` is log-transformed (`log1p`) before training
   and converted back with `expm1` when scoring.
4. **Split & clean** - A 75/25 train/test split, then an `IsolationForest`
   (contamination=0.02) removes outliers from the training set only, so
   nothing about the test set leaks into that step.
5. **Models** - Three regressors are trained and compared: `XGBRegressor`,
   `RandomForestRegressor`, and an `SVR` (wrapped in its own pipeline with a
   `StandardScaler`, since SVR is sensitive to feature scale).

## Results

From `results/results.txt` (MAE and MSE are in years, after transforming
back out of log space):

| Model | R2 | MAE | MSE |
| --- | --- | --- | --- |
| XGBoost Regressor | 0.586 | 1.47 | 4.57 |
| Random Forest Regressor | 0.588 | 1.46 | 4.55 |
| Support Vector Regressor | 0.587 | 1.46 | 4.56 |

All three land in the same range — around 1.46-1.47 years of average error —
which suggests the ceiling here is set by how much these physical
measurements alone can tell you about age, not by which model you pick.

## Getting started

```bash
uv sync
uv run src/main.py
```

This prints the evaluation report and writes it to `results/results.txt`.
