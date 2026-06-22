# Machine Learning & AI Portfolio

Welcome to my Machine Learning and Artificial Intelligence portfolio. This repository bridges the gap between **fundamental, first-principles mathematics** and **production-grade, scalable AI architecture**.

The projects inside demonstrate a balanced mastery across four core pillars: framework-less mathematical design, modern deep learning engineering, alternative metaheuristic optimization, and defensive production logic.

> 📝 **Note on Datasets:** Raw datasets are not hosted directly inside this repository due to file size constraints. However, direct data download source links are provided in the comments of every `src/` folder that requires them.

---

## 🌟 Portfolio Highlights

### 1. From-Scratch Convolutional Digit Recognition Engine (`/11_convolutional_neural_network`)

* **The Core:** A 2D Convolutional Neural Network built entirely from scratch using pure **NumPy** and **SciPy** to process, downsample, and classify spatial data (MNIST dataset).
* **The Math:** Features custom valid cross-correlation layers, vectorized max pooling via multi-dimensional array reshaping, and a numerically stable Softmax probability output layer utilizing maximum-value scaling offsets to eliminate overflow/underflow errors.
* **Engineering Win:** Implements a localized, mathematical backward pass using `'full'` correlation padding to perfectly reverse spatial feature shrinkage, allowing the entire model to backpropagate errors and learn using framework-less gradient descent.

### 2. Character-Level Sequence Modeler (`/13_name_origin_rnn`)

* **The Core:** A sequence modeling project built in **PyTorch** that reads text sequences letter-by-letter to classify the language of origin for names across 18 distinct global classes.
* **The Architecture:** Bypasses heavy tokenizers and pre-trained embeddings by processing text at the character level. Utilizes a custom Unicode-to-ASCII normalization layer (`speak_merican`) to strip complex diacritics and handle messy, real-world data safely.
* **Engineering Win:** Implements a tailored `DataLoader` using specialized item-by-item forward iterations to process dynamic sequence lengths seamlessly without requiring uniform padding overhead, protected by explicit gradient clipping to stabilize training.

### 3. Genetically Superior Mario (`/08_mario_genetics`)

* **The Core:** A reinforcement-learning-adjacent simulation that trains autonomous agents to navigate and beat complex, custom platformer environments using an evolutionary **Genetic Algorithm (GA)** in Pygame.
* **The Architecture:** Outlines agent behaviors as vectorized chromosome arrays across a fixed frame ceiling, processing fitness generations through customized tournament selections, 5-segment chronological crossovers, and localized bit-flip attribute shifts.
* **Engineering Win:** Implements dynamic, adaptive mutation boundaries. If the generation's maximum fitness metric stalls for over 5 consecutive frames, the system spikes the mutation rate by 250% to inject behavioral diversity and aggressively shatter local optima stagnation.

### 4. Kinematic Hyperparameter Optimization Swarm (`/09_hyperparameter_tuning_PSO`)

* **The Core:** A metaheuristic optimization engine that utilizes **Particle Swarm Optimization (PSO)** from scratch to navigate complex continuous search spaces and tune an unsupervised Gaussian Mixture Model (GMM) pipeline.
* **The Architecture:** Models a decentralized topology of 50 processing particles tracking personal best ($p_{best}$) and global best ($g_{best}$) coordinates across multi-dimensional continuous hyperspaces to minimize Bayesian Information Criterion (BIC) scores.
* **Engineering Win:** Solves non-convex, mixed-type parameter optimization problems without relying on slow grid searches. Employs decaying inertia parameters ($w = 0.9 \to 0.4$) for exploration/exploitation balance and safely intercepts unstable data configurations by throwing fallback penalty scores (`inf`) to guide the swarm away from mathematical runtime exceptions.

---

## 🧭 Project Index (Chronological)

| # | Project Domain | Core Methodology / Models | Tech Stack | Key Focus / Takeaway |
| --- | --- | --- | --- | --- |
| **1** | **Foundational ML** | Single-Layer Perceptron (From Scratch) | NumPy, Python | Mathematical optimization & linear boundary limits. |
| **2** | **Tabular Optimizations** | HistGradientBoosting Regressor | Scikit-Learn, Pandas | Isolated, leakage-proof transformer pipelines. |
| **3** | **Anomaly Detection** | XGBoost Classification | XGBoost, Scikit-Learn | Cyclical temporal mapping ($sin$/$cos$) for fraud. |
| **4** | **Advanced Regression** | XGBoost, Random Forest, SVR | Scikit-Learn, XGBoost | Outlier isolation using unsupervised Isolation Forests. |
| **5** | **Ensemble Classifiers** | Consensus Ensemble (8 Models) | Scikit-Learn, Python | Asymmetric preprocessing & risk-averse voting pipelines. |
| **6** | **Spatial Forecasting** | XGBoost Classifier & Regression | XGBoost, FastAPI | Macro-stratified geographic data splitting. |
| **7** | **Customer Intelligence** | Gaussian Mixture Model (GMM) | Scikit-Learn, FastAPI | Clustering transaction histories with RobustScaling. |
| **8** | **Evolutionary Computing** | Genetic Algorithm (GA) | NumPy, Pygame | Vectorized chromosomes & adaptive mutation boundaries. |
| **9** | **Metaheuristic Optimization** | Particle Swarm Optimization (PSO) | NumPy, Scikit-Learn | Continuous kinematic search to tune GMM space. |
| **10** | **Deep Learning** | Multi-Head NN (Scratch & PyTorch) | NumPy, PyTorch, Vercel | Dual-track framework testing & serverless API deployment. |
| **11** | **Computer Vision** | 2D Convolutional NN (From Scratch) | NumPy, SciPy, OpenCV | Spatial feature-map extraction & custom backward pass. |
| **12** | **Deep CV Pipelines** | Convolutional Neural Network (CNN) | PyTorch, Matplotlib | Universal hardware runtimes (`CUDA`/`MPS`), dropout masks, and interactive training loops. |
| **13** | **Natural Language Processing** | Character-Level Recurrent NN (RNN) | PyTorch, Unicodedata | Text tokenization from scratch, sequence handling, and cross-entropy classification. |

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
cd 13_name_origin_rnn

```

3. Run the core pipeline:

```bash
python main.py

```
