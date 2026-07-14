<div align="center">

# 🌸 Iris Perceptron

**A single-layer perceptron, built by hand in NumPy — no scikit-learn.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

</div>

---

Classifies one Iris species from the classic Iris dataset. The point was to implement the weight-update rule by hand and see where a linear model breaks down.

## 🧠 How it works

1. **Load the data** — `data/iris.csv` is loaded, duplicates and nulls are dropped, and the species column is one-hot encoded. Only the `virginica` column is kept as the target, making this binary: "virginica or not?"
2. **Features** — `sepal_length`, `sepal_width`, `petal_length`, and `petal_width` are the four inputs.
3. **The perceptron** — `src/perceptron.py` is plain NumPy. Weights start at zero and each sample nudges them by the classic rule: `learning_rate * error * input`.
4. **Training** — `main.py` trains on the full dataset (no train/test split — this is a from-scratch exercise, not a benchmark) for up to 1000 epochs, stopping early only on a zero-misclassification pass.

## 📊 Results

Virginica isn't perfectly linearly separable from the other species on these four features, so the perceptron never fully converges — it runs all 1000 epochs. From `results/results.txt`:

| Metric | Value |
| --- | --- |
| Dataset size | 147 records |
| Epochs completed | 1000 (did not converge) |
| Final training accuracy | **0.9796** |
| Learned weights | `[-0.998, -1.249, 1.553, 2.463]` |
| Learned bias | `-1.7900` |

`results/prediction_comparison.png` plots ground-truth vs. predicted labels side by side on the sepal dimensions.

## 🚀 Getting started

```bash
uv sync
uv run main.py
```

Writes `results/results.txt` and `results/prediction_comparison.png`.
