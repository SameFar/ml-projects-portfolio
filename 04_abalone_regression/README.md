# Abalone Physical Measurement Age Predictor

### 🎯 The Quirk & Learning Objective
Predicting the age of an abalone traditionally requires a tedious physical process: cutting open the shell, staining it, and counting the rings under a microscope. 

The core objective of this project was to determine whether **non-invasive physical dimensions** (weights, lengths, ratios) can accurately predict age using regression models. 

**Key engineering patterns implemented:**
* **Target Variance Reduction:** The target feature (`Age`) exhibits a noticeable right skew. Applied a log transformation ($y = \log(1 + x)$) to regularize variances and transformed back post-prediction via exponential evaluation.
* **Leakage Prevention Outlier Isolation:** Applied an unsupervised `IsolationForest` model to clean up anomalies, ensuring fitting was isolated **strictly to the training subset** to avoid target data leakage.
* **Dummy Variable Trap Mitigation:** Dropped the reference intercept category (`Sex_I`) during categorical one-hot encoding to preserve structural matrix independence.

### 🛠️ Feature Engineering Strategy
I built out domain-informed, complex interactive features from standard baseline inputs:
* **Approximated Volume:** `Length` × `Diameter` × `Height`
* **Bulk Density Metric:** `Whole weight` / `Volume`
* **Edible Yield Ratios:** `Shucked weight` / `Whole weight` and `Shell weight` / `Whole weight`

### 📊 Benchmark Matrix Results
After performing standard hyperparameter configuration tuning via `RandomizedSearchCV`, the multi-family model run logged the following scores:

| Model Paradigm | R² Score | Mean Absolute Error (MAE) | Mean Squared Error (MSE) |
| :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **0.590** | **1.46 years** | **4.52** |
| **Random Forest Regressor** | 0.588 | 1.46 years | 4.55 |
| **Support Vector Regressor (RBF)** | 0.587 | 1.46 years | 4.56 |

### 💡 Engineering Conclusions
* **Algorithmic Convergence:** Despite fundamentally different mathematical execution styles, all three model families capped out near an identical score threshold (**~1.46 years MAE**). This implies the predictive ceiling is constrained by structural limits within physical attribute correlations rather than variance limitations inside specific model structures.
* **Scaling Requirements:** The custom feature scaling pipeline was explicitly engineered inside an isolated `Pipeline` module for the Support Vector Machine framework, safeguarding performance against variance in unscaled engineered volumetric factors.

### 🚀 Running the Production Pipeline
Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder. To run the automated workflow from data collection to metric file parsing:
```bash
uv sync
uv run src/main.py