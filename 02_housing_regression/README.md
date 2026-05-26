# California Housing Price Prediction

Regression pipeline to predict median house prices using the California Housing dataset.

## What I Built

- Scikit-learn pipeline with proper train/test isolation
- IQR-based outlier filtering
- Quantile transformation for feature normalization
- Log transformation on target to stabilize variance
- Histogram-based Gradient Boosting model

## Results

- Full metrics stored in `results/results.txt`
- Evaluation includes R² and RMSE

## Structure

- `HouseLearning.ipynb` → EDA + experiments
- `src/` → data, model, and training pipeline
- `results/` → logs + prediction plots

## Run

```bash
pip install -r requirements.txt
python src/main.py
