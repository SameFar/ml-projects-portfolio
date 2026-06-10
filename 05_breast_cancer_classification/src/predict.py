# src/predict.py
import os
import pickle
import pandas as pd
from typing import Any

# The exact 25-feature matrix the models expect
EXPECTED_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'radius_se', 'perimeter_se',
    'area_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def load_all_models(models_dir: str | os.PathLike) -> dict[str, Any]:
    model_names = ["SVC", "KNN", "Logistic_Regression", "Naive_Bayes", "Random_Forest", "Decision_Tree", "XGBoost", "SGD"]
    models_dict = {}
    
    for name in model_names:
        file_path = os.path.join(models_dir, f"{name}_model.pkl")
        with open(file_path, "rb") as f:
            display_name = name
            models_dict[display_name] = pickle.load(f)
            
    return models_dict

def run_ensemble_vote(sample_df: pd.DataFrame, models_dict):
    # Force exact column alignment and ordering to prevent matrix dimension mismatch
    X = sample_df[EXPECTED_FEATURES]
    
    predictions = {}
    for name, model in models_dict.items():
        predictions[name] = int(model.predict(X)[0])
        
    # Calculate the total consensus vote count
    total_malignant_votes = sum(predictions.values())
    
    # Clinical safety threshold: > 3 flags malignant if at least 4 out of 8 models agree
    decision = "Malignant" if total_malignant_votes > 3 else "Benign"
    
    return decision, total_malignant_votes, predictions

if __name__ == "__main__":
    # Setup paths relative to script location
    base_dir = os.path.dirname(__file__)
    models_directory = os.path.join(base_dir, "../saved_models")
    sample_data_path = os.path.join(base_dir, "../data.csv") # Adjust if your test csv is located elsewhere
    
    print("--- Loading Serialized Production Models ---")
    models = load_all_models(models_directory)
    
    print(f"--- Processing Inbound Sample from {os.path.basename(sample_data_path)} ---")
    raw_data = pd.read_csv(sample_data_path)
    sample_row = raw_data.sample(1, random_state=42) # Fixed seed for reproducible output checks
    
    decision, votes, individual_preds = run_ensemble_vote(sample_row, models)
    
    print("\n==========================================")
    print(f"ENSEMBLE DECISION: {decision}")
    print(f"CONSENSUS TRACKER: {votes}/8 Models Flagged Malignant")
    print("==========================================\n")
    
    for model_name, pred in individual_preds.items():
        label = "Malignant (1)" if pred == 1 else "Benign (0)"
        print(f" -> {model_name:<20}: {label}")