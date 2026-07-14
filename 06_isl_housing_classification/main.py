import os
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

from src import get_pipelines, load_and_clean_data, run_macro_stratified_split


def main():
    data_path = Path(__file__).resolve().parent / "data" / "isl_housing.csv"

    print("Executing localized property parsing pipeline...")
    processed_df = load_and_clean_data(data_path)
    X_train, X_test, y_train, y_test = run_macro_stratified_split(processed_df)

    # Establish base pipeline wrapper
    pipe = get_pipelines()
    base_pipeline = pipe["XGBClass"]

    base_pipeline.fit(X_train, y_train)
    test_predictions = base_pipeline.predict(X_test)

    print("\n=== Held-Out Out-of-Sample Performance Profile ===")
    print(
        f"Validation Target Set Accuracy: {accuracy_score(y_test, test_predictions):.4f}"
    )
    print(classification_report(y_test, test_predictions))

    # Path-independent absolute structural mapping
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "saved_models")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_artifact_path = os.path.join(OUTPUT_DIR, "housing_xgb_pipeline.pkl")

    # Package model weights and exact column tracking indices together
    production_payload = {
        "pipeline": base_pipeline,
        "features": X_train.columns.tolist(),
    }

    print(f"Serializing production pipeline asset to: {output_artifact_path}")
    with open(output_artifact_path, "wb") as f:
        pickle.dump(production_payload, f)

    print("Export complete. System ready for inference deployment.")


if __name__ == "__main__":
    main()
