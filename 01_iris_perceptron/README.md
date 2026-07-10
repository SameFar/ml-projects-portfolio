# Iris Perceptron

A single-layer Perceptron built from scratch with NumPy — no scikit-learn — to
classify one Iris species from the classic Iris dataset. The point of this one
was to implement the actual weight-update rule by hand and see where a linear
model like this breaks down.

## How it works

1. **Load the data** - `data/iris.csv` is loaded, duplicates and nulls are
   dropped, and the species column is one-hot encoded. Only the `virginica`
   column is kept as the target, so this becomes a binary classification
   problem: "is this flower virginica or not?"
2. **Features** - `sepal_length`, `sepal_width`, `petal_length`, and
   `petal_width` are used as the four inputs.
3. **The Perceptron** - `src/perceptron.py` is a plain NumPy implementation.
   Weights start at zero, and each sample updates them with the classic
   perceptron rule: predict, compare to the true label, and nudge the weights
   and bias by `learning_rate * error * input`.
4. **Training** - `src/main.py` trains on the entire dataset (there's no
   train/test split here — this is a from-scratch exercise, not a benchmark)
   for up to 1000 epochs, stopping early only if a full epoch passes with zero
   misclassifications.

## Results

Virginica isn't perfectly linearly separable from the other two species on
just these four features, so the perceptron never fully converges — it runs
all 1000 epochs. From `results/results.txt`:

- Dataset size: 147 records
- Epochs completed: 1000 (did not converge)
- Final training accuracy: 0.9796
- Learned weights: `[-0.998, -1.249, 1.553, 2.463]`
- Learned bias: `-1.7900`

`results/prediction_comparison.png` plots ground-truth vs. predicted labels
side by side on the sepal dimensions.

## Getting started

```bash
uv sync
uv run src/main.py
```

This writes `results/results.txt` and `results/prediction_comparison.png`.
