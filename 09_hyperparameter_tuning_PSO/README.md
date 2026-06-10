# PSO Hyperparameter Optimizer for GMM

A custom, from-scratch implementation of a **Particle Swarm Optimization (PSO)** algorithm built to tune hyperparameters for a Gaussian Mixture Model (GMM).

## Overview

This project implements a continuous metaheuristic optimization algorithm to find the ideal configuration for an unsupervised GMM pipeline. Instead of relying on slow brute-force grid searches, a swarm of 50 particles navigates a multi-dimensional search space to minimize the Bayesian Information Criterion (BIC) score.

The optimization pipeline handles three mixed-type hyperparameters:

* **`n_components`** (Discrete integer): Number of clusters to find ($2$ to $20$).
* **`covariance_type`** (Categorical mapped to continuous space): The geometric shape of the clusters (`full`, `tied`, `diag`, `spherical`).
* **`tol`** (Continuous float): Convergence tolerance threshold ($0.0001$ to $0.1$).

---

## Algorithm Architecture

The optimization engine uses a system of physical kinematics applied to data science hyperparameter spaces. Particles adjust their movement based on momentum, personal history, and the collective wisdom of the swarm.

### Mathematical Parameters

* **Decaying Inertia ($w$):** Dynamically scales from $0.9$ down to $0.4$ across iterations to transition the swarm from wide-area exploration to tight-area exploitation.
* **Acceleration Coefficients ($c_1, c_2$):** Set to $1.8$ to evenly balance a particle's pull between its own personal best position (`pBest`) and the global swarm's best position (`GsBest`).
* **Velocity Clamping (`V_MAX`):** Restricts maximum steps to 20% of the parameter ranges to prevent particles from flying outside the boundaries.

```python
# The Core Kinematic Velocity Update Vector
v1 = (w * particle.V) + (c1 * r1 * (particle.pBest - particle.X)) + (c2 * r2 * (GpBest - particle.X))

```

---

## Search Space Mapping

Because PSO operates naturally in a continuous floating-point space, the algorithm clips and transforms vectors into strict parameter boundaries before running fitness evaluations:

```python
n_comp = int(np.clip(round(self.X[0]), L_BOUND[0], U_BOUND[0]))
cov_idx = int(np.clip(round(self.X[1]), L_BOUND[1], U_BOUND[1]))
tol = round(float(np.clip(self.X[2], L_BOUND[2], U_BOUND[2])), 5)

```

If a particle lands on an unstable configuration that causes a mathematical exception during GMM fitting, the pipeline catches the error and assigns a fitness score of `float('inf')`, cleanly forcing the swarm away from invalid search regions.

---

## Usage

### Run Optimizer

Make sure your target customer dataset is exported as a `data.csv` file in the src folder, then run the main entry point:

```bash
python main.py

```

### Outputs

The script updates live convergence progress in the terminal window and displays the winning GMM parameters at completion:

```text
Initial Swarm Best BIC: 245120.44

1/50, Best BIC score: 231104.12
2/50, Best BIC score: 228945.87
...
50/50, Best BIC score: 214052.31

Best parameters for Gaussian Mixture
            n_components = 14,
            covariance_type = full,
            tol = 0.00450

```

---

## Tech Stack

* Python
* NumPy (Vectorized kinematic calculations)
* Pandas
* Scikit-learn (GaussianMixture evaluation)
* Pathlib

---