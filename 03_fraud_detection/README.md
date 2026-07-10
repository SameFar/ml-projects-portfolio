# Credit Card Fraud Detection

An XGBoost classifier that flags fraudulent credit card transactions in the
[Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Fraud is a tiny fraction of the data (492 out of roughly 284,000
transactions), so the interesting part of this project is handling that
imbalance rather than the model itself.

## How it works

1. **Load the data** - `data/creditcard.csv` is loaded, deduplicated, and
   cleaned of nulls.
2. **Feature engineering**
   - `Time` (seconds since the first transaction) is converted to an hour of
     day, then encoded as `Hour_sin` / `Hour_cos` so the model sees hour 23
     and hour 0 as adjacent instead of far apart.
   - `Amount` is log-transformed (`LogAmount`) to tame its skew.
   - `Time`, `Amount`, `Hour`, and the PCA components `V13`, `V15`, `V20`,
     `V22`, `V23`, `V25`, `V28` are dropped — an exploratory KDE comparison
     (see `FraudDetection.ipynb`) showed the fraud/non-fraud distributions on
     those columns basically overlap, so they're not adding signal.
3. **Model** - A `Pipeline` of `RobustScaler` (less sensitive to the extreme
   outliers fraud creates) followed by an `XGBClassifier`
   (`learning_rate=0.15`, `max_depth=7`, `n_estimators=1000`).
4. **Split & evaluate** - An 80/20 train/test split, then precision, recall,
   and a confusion matrix on the held-out set.

`src/model.py` also defines a `GridSearchCV` wrapper (`get_grid_search`) that
tunes over `n_estimators`, `max_depth`, and `learning_rate`, tracking
precision but refitting on recall — but `src/main.py` currently only runs the
untuned baseline pipeline, not the grid search.

There's a second notebook, `FraudDetection-Isolation.ipynb`, exploring an
Isolation Forest approach to the same problem, separate from the XGBoost
pipeline `src/main.py` runs.

## Results

From `results/results.txt`, on 56,746 held-out transactions:

- Precision: 0.9701
- Recall: 0.7222
- True negatives: 56,654
- False positives: 2
- False negatives (missed fraud): 25
- True positives (fraud caught): 65

`results/confusion_matrix.png` has the heatmap version.

## Getting started

You'll need `data/creditcard.csv` from the
[Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
first.

```bash
uv sync
uv run src/main.py
```

This writes `results/results.txt` and `results/confusion_matrix.png`.
