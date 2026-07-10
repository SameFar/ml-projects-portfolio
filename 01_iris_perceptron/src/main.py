import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_and_preprocess
from perceptron import Perceptron
from pathlib import Path


def main():
    # dataset from https://www.kaggle.com/datasets/vikrishnan/iris-dataset
    data_path = Path(__file__).resolve().parent.parent / "data" / "iris.csv"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Prepare data
    df, X_data, y_data = load_and_preprocess(data_path)
    nums = df.shape[0]

    # 2. Train model
    P = Perceptron(inputs=4, learning_rate=0.01)
    epochs = 1000
    final_epoch = epochs
    converged = False

    for epoch in range(epochs):
        total_error = 0
        for i in range(nums):
            error = P.train(X_data[i], y_data[i])
            total_error += abs(error)

        if total_error == 0:
            final_epoch = epoch
            converged = True
            break

    # 3. Predict & evaluate
    df["predicted_target"] = [P.activate(x) for x in X_data]
    df["predicted_species"] = df["predicted_target"]

    accuracy = (df["predicted_species"] == df["virginica"]).sum() / nums

    # 4. Output metrics to results.txt
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write("=== Perceptron From Scratch Run Info ===\n")
        f.write(f"Total dataset size: {nums} records\n")
        f.write(f"Epochs completed: {final_epoch}\n")
        f.write(f"Converged perfectly: {converged}\n")
        f.write(f"Final training accuracy: {accuracy:.4f}\n")
        f.write(f"Learned Weights: {P.weights}\n")
        f.write(f"Learned Bias: {P.bias:.4f}\n")

    # 5. Generate and save the visual comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        ax=axes[0], x="sepal_length", y="sepal_width", data=df, hue="virginica"
    )
    axes[0].set_title("Ground Truth")

    sns.scatterplot(
        ax=axes[1], x="sepal_length", y="sepal_width", data=df, hue="predicted_species"
    )
    axes[1].set_title("Perceptron Predictions")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prediction_comparison.png"))
    plt.close()

    print(
        f"Run completed. Metrics and plots exported to '{results_dir}/'. Accuracy: {accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
