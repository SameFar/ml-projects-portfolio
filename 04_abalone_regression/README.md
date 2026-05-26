# Abalone Age Predictor

Predicting abalone age normally requires cutting the shell and counting rings manually.  
This project predicts age using only physical measurements with regression models.

## What I Implemented

- Log transformation on the target to reduce skew
- Leakage-safe outlier removal using `IsolationForest`
- Feature engineering using volume, density, and weight ratios
- Hyperparameter tuning with `RandomizedSearchCV`
- Multiple regression model benchmarks

## Best Result

- XGBoost Regressor
- MAE: `1.46 years`
- R² Score: `0.59`

Full experiment outputs are available in `results.txt`.

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy

## Run

```bash
python src/train.py
