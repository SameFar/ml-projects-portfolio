# Multi-Paradigm Breast Cancer Classification & Consensus Ensemble

An end-to-end machine learning pipeline implementing a heterogeneous threshold-consensus ensemble across **8 distinct algorithmic families**. This repository demonstrates production-grade engineering patterns, including asymmetric preprocessing pipelines, automated feature selection, and a clinical-grade risk-averse inference engine.

---

## 🏗️ Architecture Design & Engineering Patterns

Rather than applying a generic blanket pipeline, the system utilizes a split-architecture approach to optimize data structures based on mathematical assumptions of the underlying model families:

* **Asymmetric Preprocessing Pipelines:** * **Distance & Gradient-Based Models** (*Logistic Regression, SVC, SGD, KNN, Naive Bayes*): Data is dynamically transformed via a `Yeo-Johnson PowerTransformer` to minimize skew, followed by an `IsolationForest` wrapper to dynamically prune extreme leverage points.
  * **Tree-Based Models** (*Decision Trees, Random Forests, XGBoost*): Processed using the raw input distribution, preserving data splitting criteria and avoiding unnecessary computation on scale-invariant architectures.
* **Dynamic Feature Selection:** The inference pipeline enforces strict input validation against a 25-dimensional structural measurement matrix. It programmatically drops low-correlation attributes ($|r| < 0.01$) dynamically before processing to reduce feature noise and mitigate the curse of dimensionality.
* **Clinical Sensitivity Tuning (Ensemble Vote):** To simulate a real-world clinical decision-support system, the inference engine replaces standard majority voting with a conservative risk-averse threshold (`vote > 3`). If 4 or more of the 8 models flag malignancy, the final output forces a **Malignant** classification—heavily penalizing false negatives to optimize patient safety in screening diagnostics.

---

## 📊 Performance Metric Summary

Inside results folder

### 💡 Core Engineering Takeaways
1. **Occam's Razor in High-Dimensional Space:** Regularized Logistic Regression outperformed complex tree ensembles. This indicates that once heavy feature skewness is mathematically stabilized, the underlying decision boundary is predominantly linear.
2. **Consensus Regularization:** While standalone tree models underperformed due to variance, their integration into the conservative consensus wrapper acts as an algorithmic buffer, catching edge cases where distance estimation models hit margin limitations.

---

## 🚀 Deployment & Execution Guide

### 1. Training Pipeline
To execute the end-to-end training pipeline, optimize hyperparameters, and serialize the binary weights (`.pkl`/`.onnx` format):

Bash
python src/train.py

### 2. Production Inference & Prediction
Before executing the prediction file, you must edit data.csv and populate it with real, unstructured patient values matching the 25 required features (e.g., radius_mean, texture_worst, etc.) for the engine to evaluate.

Bash
python src/predict.py --input data.csv