# Project Name: California Housing Price Regression via Scikit-Learn Pipelines

A modular regression project utilizing an optimized scikit-learn Pipeline structure to predict median house values using the California Housing dataset. 

## 💡 The Quirk & Learning Objective
For this milestone in my portfolio, **the core emphasis was mastering complex feature transformations and deploying scikit-learn's native Histogram-Based Gradient Boosting algorithm while preventing data leakage.**

Key elements focused on during this iteration:
* **Robust Table Cleaning:** Implemented an automated global Interquartile Range (IQR) masking routine to strip erratic outlier records out of continuous feature sets.
* **Leakage-Proof Pipeline Construction:** Used an encapsulation pipeline to pass variables through a non-linear `QuantileTransformer` (mapping outputs to a gaussian distribution) ensuring data transformations are isolated strictly to cross-validation splits.
* **Logarithmic Target Scaling:** Addressed variance skew by applying a mathematical $y = \log(1 + x)$ adjustment to stabilize gradient steps across regression sweeps.

## 📁 Repository Structure
* `HouseLearning.ipynb`: Exploration workbook demonstrating baseline metrics checking, outlier boundaries evaluation, and pipeline assembly verification.
* `src/`: Isolated production environment components.
  * `data_loader.py`: Fetches dataset variables, handles IQR removal masks, and splits matrices.
  * `model.py`: Sets up pipeline definitions and parameter configurations.
  * `main.py`: Handles model execution, records key outputs to files, and builds verification diagrams.
* `results/`: Tracking records asset directory.
  * `results.txt`: Performance tracking sheets listing $R^2$ benchmarks and Root Mean Squared Error (RMSE) readouts.
  * `prediction_analysis.png`: Scatter plot tracking actual log pricing values against predicted trends.

## 📊 Key Results & Insights
* **The Structural Advantage of HistGradientBoosting:** Native histogram binning handles large tabular features significantly faster than conventional decision tree structures, yielding quick iterations during pipeline passes.
* **Transformation Gains:** Normalizing feature scaling distributions using a standard quantile map prevented heavily skewed features (like population counts or room metrics) from skewing weight updates.
* **Terminal Summary:** The completed run achieved an $R^2$ coefficient of **0.XX** (refer to your updated `results/results.txt` for your precise machine output parameters).

## 🚀 Execution & Portability
This repository relies exclusively on relative path lookups and contains zero environment-specific dependencies. 

Install target packages:
```bash
pip install -r requirements.txt