# Machine Learning & AI Portfolio

Welcome to my Machine Learning and Artificial Intelligence portfolio. This repository bridges the gap between **fundamental, first-principles mathematics** and **production-grade, scalable AI architecture**.

---

## 🌟 Portfolio Highlights

### 1. Multi-Head Fashion Recommendation Network (`/10_neural_network`)

* **The Core:** A neural network built to predict multiple fashion recommendations simultaneously (colors, patterns, materials).
* **The Quirk:** Contains **two separate tracks**: a from-scratch **NumPy** version to prove backpropagation calculus math, and a modern **PyTorch** version for industry-standard speed.
* **Production Win:** The framework-less NumPy weights are deployed on **Vercel Serverless**, completely bypassing the heavy file-size limits and cold-start delays of massive ML libraries.

### 2. Multi-Paradigm Breast Cancer Consensus Ensemble (`/05_breast_cancer_classification`)

* **The Core:** A clinical-grade diagnostic engine combining **8 distinct algorithmic families** (Trees, Linear Models, Distance Models) into one robust ensemble.
* **Production Win:** Implements a risk-averse threshold voting strategy ($vote > 3$). If 4 or more models flag a risk, the system forces a malignant diagnosis—heavily penalizing dangerous false negatives to ensure patient screening safety.

### 3. Genetically Superior Mario (`/08_mario_genetics`)

* **The Core:** A reinforcement-learning-adjacent simulation that trains autonomous agents to beat custom platformer levels using an evolutionary **Genetic Algorithm (GA)** in Pygame.
* **Production Win:** Features adaptive mutation rates. If the generation's maximum fitness stalls for over 5 frames, mutation spikes by 250% to inject behavioral diversity and aggressively break out of local optima.

---

## 🧭 Project Index (Chronological)

| # | Project Domain | Core Methodology / Models | Tech Stack | Key Focus / Takeaway |
| --- | --- | --- | --- | --- |
| **1** | **Foundational ML** | Single-Layer Perceptron (From Scratch) | NumPy, Python | Mathematical optimization & linear boundary limits. |
| **2** | **Tabular Optimizations** | HistGradientBoosting Regressor | Scikit-Learn, Pandas | Isolated, leakage-proof transformer pipelines. |
| **3** | **Anomaly Detection** | XGBoost Classification | XGBoost, Scikit-Learn | Cyclical temporal mapping ($sin$/$cos$) for fraud. |
| **4** | **Advanced Regression** | XGBoost, Random Forest, SVR | Scikit-Learn, XGBoost | Outlier isolation using unsupervised Isolation Forests. |
| **5** | **Ensemble Classifiers** | Consensus Ensemble (8 Models) | Scikit-Learn, Python | Asymmetric preprocessing & risk-averse voting. |
| **6** | **Spatial Forecasting** | XGBoost Classifier & Regression | XGBoost, FastAPI | Macro-stratified geographic data splitting. |
| **7** | **Customer Intelligence** | Gaussian Mixture Model (GMM) | Scikit-Learn, FastAPI | Clustering transaction histories with RobustScaling. |
| **8** | **Evolutionary Computing** | Genetic Algorithm (GA) | NumPy, Pygame | Vectorized chromosomes & adaptive mutation boundaries. |
| **9** | **Metaheuristic Optimization** | Particle Swarm Optimization (PSO) | NumPy, Scikit-Learn | Continuous kinematic search to tune GMM space. |
| **10** | **Deep Learning (Highlight)** | Multi-Head NN (Scratch & PyTorch) | NumPy, PyTorch, Vercel | Dual-track framework testing & serverless API. |

---

## 🚀 Getting Started

Every project folder in this repository is completely self-contained with relative pathing and separate environments to eliminate package conflicts.

To run any module:


1. Setup dependencies:
```bash
pip install -r requirements.txt

```

2. Navigate to the target project directory:
```bash
cd 01-iris_perceptron

```

3. Run the core pipeline:
```bash
python src/main.py

```
