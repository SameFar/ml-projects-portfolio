# Multi-Paradigm Breast Cancer Classification & Consensus Ensemble

### 🎯 The Quirk & Learning Objective
This project implements a multi-model evaluation strategy to study how **8 distinct algorithmic families** react to variance-stabilizing feature scaling and unsupervised outlier filtration. 

Beyond individual model exploration, the second core learning objective was to build a **heterogeneous threshold-consensus ensemble script** that aggregates predictions across all 8 architectures to simulate a clinical decision-support system.

**Key engineering patterns implemented:**
* **Asymmetric Preprocessing Pipelines:** 
  * *Parametric Models* (Logistic Regression, SVC, SGD, KNN, Naive Bayes) process data optimized via a `Yeo-Johnson PowerTransformer` and an `IsolationForest` wrapper to drop extreme leverage points.
  * *Non-Parametric Models* (Trees, Random Forests, XGBoost) process the raw training distribution directly, as they are naturally scale-invariant.
* **Clinical Sensitivity Tuning (Ensemble Vote):** Instead of a standard majority vote threshold (> 4), the inference engine implements a conservative threshold rule (`vote > 3`). If 4 or more models flag malignancy, the final system output forces a **Malignant** classification. This architecture penalizes false negatives heavily, making it far safer for screening diagnostics.

### 🛠️ Production Feature Matrix
The inference pipeline explicitly checks, structures, and enforces input data against these 25 structural measurements, completely stripping uninformative low-correlation attributes ($|r| < 0.01$) dynamically before processing:
`radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, and `concave points` across their respective `_mean`, `_se`, and `_worst` variations.

### 📊 Performance Metric Summary
The individual validation runs logged the following competitive benchmarks:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Preprocessing Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.982** | **0.982** | **0.982** | **0.982** | Yeo-Johnson + Isolation Forest |
| **SGD Classifier** | 0.974 | 0.974 | 0.974 | 0.974 | Yeo-Johnson + Isolation Forest |
| **Support Vector Classifier (SVC)** | 0.974 | 0.974 | 0.974 | 0.974 | Yeo-Johnson + Isolation Forest |
| **K-Nearest Neighbors (KNN)** | 0.965 | 0.965 | 0.965 | 0.965 | Yeo-Johnson + Isolation Forest |
| **Gaussian Naive Bayes** | 0.956 | 0.956 | 0.956 | 0.956 | Yeo-Johnson + Isolation Forest |
| **Random Forest** | 0.956 | 0.956 | 0.956 | 0.956 | Raw Input Distribution |
| **XGBoost** | 0.956 | 0.956 | 0.956 | 0.956 | Raw Input Distribution |
| **Decision Tree** | 0.921 | 0.922 | 0.921 | 0.921 | Raw Input Distribution |

### 💡 Core Takeaways
* **Linear Simplicity Wins:** For this high-dimensional feature space, a regularized Logistic Regression model outpaced complex tree ensembles. This strongly suggests that the underlying class decision boundary is mostly linear once heavy skewness is stabilized.
* **The Benefit of Consensus:** While individual trees or naive bayes models score lower standalone, when tied together into the consensus wrapper, they act as stable regularizers, catching edge cases where margins might clip in distance estimation algorithms.

### 🚀 Running the Production Pipeline

**To retrain all models and re-export the binary weights:**
```bash
python src/train.py