# Perceptron from Scratch (Iris - Virginica)

Single-layer Perceptron built from scratch using NumPy for binary classification on Iris Virginica.

## What I Built

- Perceptron implemented without ML libraries
- Manual weight update rule using gradient-style error correction
- Training loop with fixed epoch cap
- Basic evaluation + visualization pipeline

## Key Insight

- Data is not fully linearly separable
- No guaranteed convergence, training stabilizes after many epochs
- Accuracy ~99.7% (see `results/results.txt`)

## Structure

- `portfolio_notebook.ipynb` → experimentation
- `src/perceptron.py` → core model
- `src/data_loader.py` → dataset prep
- `results/` → logs + plots

## Run

```bash
cd src
python train.py
