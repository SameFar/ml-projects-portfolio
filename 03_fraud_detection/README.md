<div align="center">

# 💳 Credit Card Fraud Detection

**XGBoost against a 0.17%-positive class — the imbalance is the real problem.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

</div>

---

Flags fraudulent transactions in the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Fraud is 492 of ~284,000 transactions, so the interesting part is handling that imbalance, not the model.

## 🧠 How it works

1. **Load** — `data/creditcard.csv` is loaded, deduplicated, and cleaned of nulls.
2. **Feature engineering**
   - `Time` → hour of day, encoded as `Hour_sin` / `Hour_cos` so hour 23 and hour 0 read as adjacent.
   - `Amount` is log-transformed (`LogAmount`) to tame its skew.
   - `Time`, `Amount`, `Hour`, and PCA components `V13`, `V15`, `V20`, `V22`, `V23`, `V25`, `V28` are dropped — a KDE comparison (`FraudDetection.ipynb`) showed their fraud/non-fraud distributions overlap, so they add no signal.
3. **Model** — `RobustScaler` (resilient to fraud's extreme outliers) → `XGBClassifier` (`learning_rate=0.15`, `max_depth=7`, `n_estimators=1000`).
4. **Split & evaluate** — 80/20 split, then precision, recall, and a confusion matrix on the held-out set.

`src/model.py` also has a `GridSearchCV` wrapper (`get_grid_search`) that tunes `n_estimators`/`max_depth`/`learning_rate`, refitting on recall — but `main.py` runs only the untuned baseline. A second notebook, `FraudDetection-Isolation.ipynb`, explores an Isolation Forest approach separately.

## 📊 Results

From `results/results.txt`, on 56,746 held-out transactions:

| Metric | Value |
| --- | --- |
| Precision | **0.9701** |
| Recall | 0.7222 |
| True negatives | 56,654 |
| False positives | 2 |
| False negatives (missed fraud) | 25 |
| True positives (fraud caught) | 65 |

`results/confusion_matrix.png` has the heatmap version.

## 🚀 Getting started

Grab `data/creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) first.

```bash
uv sync
uv run main.py
```

Writes `results/results.txt` and `results/confusion_matrix.png`.
