# Building a Perceptron from Scratch: Virginica Classification

A fundamental, modular implementation of a single-layer Perceptron built entirely with Python and NumPy. This project tests linear classification capabilities on a subset of the Iris dataset.

## 💡 Learning Objective
Instead of utilizing high-level APIs like scikit-learn, **the objective here was to build the underlying optimization algorithms from first principles.** 

This specific project allowed me to analyze:
* The core mathematical formulation of weight updates based on direct prediction error: $\Delta w = \eta \cdot (y - \hat{y}) \cdot x$
* The structural limitations of a single-layer neural network when applied to classes that cannot be cleanly split by a single linear hyperplane.

## 📁 Repository Structure
* `portfolio_notebook.ipynb`: Scratchpad workflow detailing initial exploratory data data structure checks, manual one-hot filtering, and interactive visualization code.
* `src/`: Refactored production modules.
  * `data_loader.py`: Handles duplicate/null dropping and formats features into a matrix structure.
  * `perceptron.py`: Outlines the custom explicit `Perceptron` class.
  * `main.py`: Oversees execution, metrics handling, and visualization pipelines.
* `results/`: Outward metrics storage.
  * `results.txt`: Log tracking final learned parameter weights, biases, and terminal training accuracy.
  * `prediction_comparison.png`: Side-by-side scatter plot comparing ground-truth targets against model outputs.

## 📊 Key Results & Insights
* **The Non-Convergence Phenomenon:** Because the Iris class distributions (specifically mapping Virginica using only sepal dimensional metrics) are not perfectly linearly separable, the Perceptron loop runs through all 1,000 maximum epochs without early convergence ($total\_error == 0$).
* **Decision Boundary Quality:** Despite the lack of perfect linear convergence, the simple architecture yields strong performance, stabilizing around **~99.7%** accuracy (check `results/results.txt` for exact run metrics).

## 🚀 How to Execute
Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder. To clean the data, train the model, log accuracy profiles, and output decision figures:
```bash
uv sync
uv run src/main.py