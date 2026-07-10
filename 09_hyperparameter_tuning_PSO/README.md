# PSO Hyperparameter Tuning for a Gaussian Mixture Model

A particle swarm optimizer, written from scratch with NumPy, that searches for good hyperparameters for a Gaussian Mixture Model instead of running a grid search over them.

## How it works

A GMM has a few hyperparameters that interact in non-obvious ways: `n_components` (how many clusters), `covariance_type` (`full`, `tied`, `diag`, or `spherical`), and `tol` (convergence tolerance). Grid search treats these independently and gets expensive fast; PSO instead treats them as coordinates in a 3D continuous space and lets a swarm of particles fly around that space looking for the combination that minimizes BIC (Bayesian Information Criterion) on the dataset.

Each particle (`src/Particle.py`) is a position vector `[n_components, covariance_type_index, tol]` plus a velocity vector. Since `covariance_type` is categorical, it's treated as an index (0-3) into `["full", "tied", "diag", "spherical"]` and rounded to the nearest integer before each fitness evaluation. Fitness is just "fit a `GaussianMixture` with these (rounded/clamped) parameters and return its BIC score" — lower is better. If a particle lands on a configuration that makes the GMM fail to fit, its fitness is set to infinity so the swarm steers away from it.

`src/main.py` runs the swarm: 50 particles for 50 iterations. Each iteration, every particle's velocity gets updated by a weighted mix of its own momentum, the pull toward its personal-best position, and the pull toward the swarm's global-best position — the standard PSO velocity update. The inertia weight decays linearly from 0.9 to 0.4 over the run, so the swarm explores broadly early on and settles down to fine-tune later. Velocity is clamped to 20% of each parameter's range so particles can't overshoot the search space in one step.

The dataset it's tuning against (`src/data.csv`) is the same engineered customer-summary table from [`07_customer_types`](../07_customer_types) — seven numeric features per customer (transaction count, total spent, return rate, etc.).

## Results

Running `uv run src/main.py` (50 particles, 50 iterations) against `src/data.csv`:

PLACEHOLDER_RESULTS

## Getting started

```bash
uv sync
uv run src/main.py
```

`src/main.py` expects a `data.csv` in the `src/` folder — it's included here, but you can point it at any numeric feature table you want to tune a GMM for. It prints the swarm's best BIC score after each iteration and prints the winning `n_components`, `covariance_type`, and `tol` at the end.
