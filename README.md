# Machine Learning & AI Portfolio

Welcome to my Machine Learning and Artificial Intelligence portfolio. This repository bridges the gap between **fundamental, first-principles mathematics** and **production-grade, scalable AI architecture**.

> 📝 **Note on Datasets:** Raw datasets are not hosted directly inside this repository due to file size constraints. However, direct data download source links are provided in the comments of every `src/` folder that requires them.

---

## 🌟 Portfolio Highlights

### 1. From-Scratch Convolutional Digit Recognition Engine (`/11_convolutional_neural_network`)

* **The Core:** A 2D Convolutional Neural Network built from scratch using pure **NumPy** and **SciPy** to process, downsample, and classify spatial data (MNIST dataset).
* **The Quirk:** Consolidates a custom 2D valid cross-correlation layer, ReLU activation mappings, matrix-flattening bridges, and a fully-connected dense layer with stable Softmax probabilities into a single unified execution loop.
* **Production Win:** Implements a localized, mathematical backward pass using `'full'` correlation padding to perfectly reverse spatial feature shrinkage, allowing the entire model to backpropagate errors and learn using framework-less gradient descent.

### 2. Multi-Head Fashion Recommendation Network (`/10_neural_network`)

* **The Core:** A neural network built to predict multiple fashion recommendations simultaneously (colors, patterns, materials).
* **The Quirk:** Contains **two separate tracks**: a from-scratch **NumPy** version to prove backpropagation calculus math, and a modern **PyTorch** version for industry-standard speed.
* **Production Win:** The framework-less NumPy weights are deployed on **Vercel Serverless**, completely bypassing the heavy file-size limits and cold-start delays of massive ML libraries.

### 3. Multi-Paradigm Breast Cancer Consensus Ensemble (`/05_breast_cancer_classification`)

* **The Core:** A clinical-grade diagnostic engine combining **8 distinct algorithmic families** (Trees, Linear Models, Distance Models) into one robust ensemble.
* **Production Win:** Implements a risk-averse threshold voting strategy ($vote > 3$). If 4 or more models flag a risk, the system forces a malignant diagnosis—heavily penalizing dangerous false negatives to ensure patient screening safety.

### 4. Genetically Superior Mario (`/08_mario_genetics`)

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
| **10** | **Deep Learning** | Multi-Head NN (Scratch & PyTorch) | NumPy, PyTorch, Vercel | Dual-track framework testing & serverless API. |
| **11** | **Computer Vision** | 2D Convolutional NN (From Scratch) | NumPy, SciPy, OpenCV | Spatial feature-map extraction & custom backward pass. |

---

## 🚀 Getting Started

Every project folder in this repository is completely self-contained with relative pathing.

To run any module:

1. Setup dependencies:

```bash
pip install -r requirements.txt

```

2. Navigate to the target project directory:

```bash
cd 11_convolutional_neural_network

```

3. Run the core pipeline:

```bash
python src/train.py

```
