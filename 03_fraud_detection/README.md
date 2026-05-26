# Credit Card Fraud Detection (XGBoost)

Imbalanced classification pipeline for detecting fraudulent transactions (0.17% positive class).

## What I Did

- Cyclical encoding for time features (sin/cos for hour)
- Removed noisy features using KDE distribution overlap analysis
- Optimized model for **recall** using custom GridSearchCV
- XGBoost-based classification pipeline

## Key Metrics

- Precision: **94.44%**
- Recall: **75.56%**

Full logs and breakdown: `results/results.txt`

## Structure

- `portfolio_notebook.ipynb` → EDA + experiments
- `src/` → training pipeline (data, model, train scripts)
- `results/` → metrics + confusion matrix

## Run

```bash
pip install -r requirements.txt
python src/train.py
