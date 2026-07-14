<div align="center">

# 🐝 PSO Hyperparameter Tuning

**A from-scratch particle swarm optimizer tuning a Gaussian Mixture Model.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

</div>

---

A particle swarm optimizer, written from scratch in NumPy, that searches for good GMM hyperparameters instead of running a grid search.

## 🧠 How it works

A GMM's hyperparameters interact in non-obvious ways: `n_components`, `covariance_type` (`full`/`tied`/`diag`/`spherical`), and `tol`. Grid search treats these independently and gets expensive fast; PSO treats them as coordinates in a 3D continuous space and lets a swarm fly around it, minimizing **BIC** (Bayesian Information Criterion).

Each particle (`src/Particle.py`) is a position `[n_components, covariance_type_index, tol]` plus a velocity. `covariance_type` is categorical, so it's an index (0–3) rounded before each fitness call. Fitness = "fit a `GaussianMixture` with these (rounded/clamped) params, return BIC" — lower is better. A config that fails to fit gets infinite fitness, steering the swarm away.

`main.py` runs 50 particles for 50 iterations. Each iteration, every velocity is a weighted mix of its own momentum, the pull toward its personal best, and the pull toward the swarm's global best — the standard PSO update. Inertia weight decays linearly 0.9 → 0.4 (explore early, fine-tune late), and velocity is clamped to 20% of each parameter's range so particles can't overshoot in one step.

The dataset (`src/data.csv`) is the same engineered customer-summary table from [`07_customer_types`](../07_customer_types) — seven numeric features per customer.

## 📊 Results

Running `uv run main.py` (50 particles, 50 iterations) against `src/data.csv`:

PLACEHOLDER_RESULTS

## 🚀 Getting started

```bash
uv sync
uv run main.py
```

`main.py` expects `src/data.csv` (included), but you can point it at any numeric feature table. It prints the swarm's best BIC each iteration and the winning `n_components`, `covariance_type`, and `tol` at the end.
