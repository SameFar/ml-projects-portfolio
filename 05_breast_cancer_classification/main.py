import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src import (
    load_cancer_dataset,
    process_and_split_features,
    get_outlier_detector,
    get_parametric_pipelines,
    get_non_parametric_models,
)


def main():
    print("Initiating execution routine...")
    data_dir = os.path.join(os.path.dirname(__file__), "data", "breastcancer.csv")
    df = load_cancer_dataset(data_dir)
    X, y = process_and_split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Isolate outliers in training subset
    iso = get_outlier_detector()
    inliers = iso.fit_predict(X_train)
    iX_train, iy_train = X_train[inliers == 1], y_train[inliers == 1]

    predictions = {}
    trained_artifacts = {}

    # Group A: Fit non-tree models using variance-stabilized, cleaned training data
    for name, pipeline in get_parametric_pipelines().items():
        pipeline.fit(iX_train, iy_train)
        predictions[name] = pipeline.predict(X_test)
        trained_artifacts[name] = pipeline

    # Group B: Fit tree models using the raw training data
    for name, estimator in get_non_parametric_models().items():
        estimator.fit(X_train, y_train)
        predictions[name] = estimator.predict(X_test)
        trained_artifacts[name] = estimator

    # Collect and compare metrics for each model
    metrics_log = []
    for name, y_pred in predictions.items():
        metrics_log.append(
            {
                "Model Architecture": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            }
        )

    df_compare = (
        pd.DataFrame(metrics_log)
        .sort_values(by="F1-Score", ascending=False)
        .reset_index(drop=True)
    )

    # Make filesystem archive locations
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Export execution log summary
    results_path = os.path.join(results_dir, "results.txt")
    df_compare.to_string(os.path.join(results_path), index=False)

    print("\n--- Pipeline Compilation Metrics ---")
    print(df_compare.to_string(index=False))

    # Serialize functional model states to disk binaries
    for name, model_obj in trained_artifacts.items():
        sanitized_name = name.replace(" ", "_")
        file_path = os.path.join(models_dir, f"{sanitized_name}_model.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(model_obj, f)

    print(f"\nWorkflow complete. Metrics saved to {results_path}.")


if __name__ == "__main__":
    main()
