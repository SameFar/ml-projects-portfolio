# Machine Learning & AI Portfolio

A collection of ML/AI projects I've built over time — some are from-scratch implementations of core algorithms (a perceptron, a CNN, a particle swarm optimizer), others are more standard pipelines using scikit-learn, XGBoost, or PyTorch, and a couple were just for fun (a genetic algorithm that learns to play a platformer, a model that guesses if you'll like a song).

Every project folder is self-contained: its own dataset (or a note on where to get it), its own `pyproject.toml` managed by [uv](https://docs.astral.sh/uv/), and its own README. Nothing is shared from the repo root, so you can clone or copy any single folder and it'll work on its own.

---

## My best work:

**Character-level RNN for name origin** ([`13_NLP_RNN`](13_NLP_RNN)) — reads a name letter-by-letter and guesses which of 18 languages it's from. A character-level RNN in PyTorch.

**Mario genetic algorithm** ([`08_mario_genetics`](08_mario_genetics)) — instead of reinforcement learning, a population of agents evolves via a genetic algorithm to get through a custom platformer level. When fitness stalls for a few generations, the mutation rate automatically ramps up to help it escape local optima.

**PSO hyperparameter tuning** ([`09_hyperparameter_tuning_PSO`](09_hyperparameter_tuning_PSO)) — a particle swarm optimizer built from scratch to tune a Gaussian Mixture Model's hyperparameters, as an alternative to grid search.

**CNN-GRU music preference model** ([`15_CNN-GRU_music_perferance`](15_CNN-GRU_music_perferance)) — predicts whether you'd like a song just from the audio, using a pretrained audio embedding model (PANN) feeding into a small GRU classifier. Comes with a Streamlit demo.

---

## Project Index

| # | Project | Approach | Stack | Notes |
| --- | --- | --- | --- | --- |
| **01** | [Iris Perceptron](01_iris_perceptron) | Single-layer perceptron, from scratch | NumPy, Pandas | Basic linear classifier, hits the limits of linear separability. |
| **02** | [Housing Regression](02_housing_regression) | HistGradientBoosting | Scikit-Learn, Pandas | California housing price regression with a leakage-proof pipeline. |
| **03** | [Fraud Detection](03_fraud_detection) | XGBoost classification | XGBoost, Scikit-Learn | Credit card fraud detection with heavy class imbalance. |
| **04** | [Abalone Regression](04_abalone_regression) | XGBoost, Random Forest, SVR | Scikit-Learn, XGBoost | Predicting abalone age, with isolation forest outlier removal. |
| **05** | [Breast Cancer Classification](05_breast_cancer_classification) | Ensemble of 8 models | Scikit-Learn, XGBoost | Consensus voting across several classifiers. |
| **06** | [Islamabad Housing Classification](06_isl_housing_classification) | XGBoost | XGBoost, FastAPI | Housing price bracket prediction with a small API on top. |
| **07** | [Customer Types](07_customer_types) | Gaussian Mixture Model | Scikit-Learn, FastAPI | Clustering e-commerce customers by purchase behavior. |
| **08** | [Mario Genetics](08_mario_genetics) | Genetic algorithm | NumPy, Pygame | See above. |
| **09** | [PSO Hyperparameter Tuning](09_hyperparameter_tuning_PSO) | Particle swarm optimization | NumPy, Scikit-Learn | See above. |
| **10** | [Fashion Neural Network](10_neural_network) | Small multi-head NN (NumPy + PyTorch) | NumPy, PyTorch, FastAPI | Fashion recommendation model, deployed to Vercel. |
| **11** | [Convolutional Neural Network](11_convolutional_neural_network) | 2D CNN, from scratch | NumPy, SciPy, OpenCV | MNIST digit classifier with a hand-written forward/backward pass. |
| **12** | [Torch CNN](12_torch_CNN) | CNN | PyTorch, Kagglehub | Intel image classification, with an interactive training loop. |
| **13** | [NLP RNN](13_NLP_RNN) | Character-level RNN | PyTorch | See above. |
| **14** | [Stock Prediction LSTM](14_stock_prediction_lstm) | LSTM | PyTorch, yfinance | Predicts next-day NVDA log-returns from a rolling window. |
| **15** | [CNN-GRU Music Preference](15_CNN-GRU_music_perferance) | Pretrained CNN embeddings + GRU | PyTorch, librosa, Streamlit | See above. |
| **16** | [CDA](16_CDA) *(work in progress)* | LLM-based contract clause retrieval | Google Gen AI SDK | Early exploration, mostly empty stubs right now. |

---

## Getting started

Each project is independent, so pick the one you want and:

```bash
cd 13_NLP_RNN
uv sync
uv run src/main.py
```

The exact run command varies by project (some use `main.py` at the root, some use `src/main.py`, a couple expose a FastAPI app) — check that project's own README for specifics, including where its dataset comes from and any extra setup (like downloading a Kaggle dataset or installing `ffmpeg`).
