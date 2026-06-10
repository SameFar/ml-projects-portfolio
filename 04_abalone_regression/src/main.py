import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_base_data, build_features
from model import clean_training_outliers, get_baseline_models, evaluate_predictions

def main():
    print("Executing pipeline...")
    
    # 1. Pipeline ingestion & cleaning
    df = load_base_data()
    X, y = build_features(df)
    
    # 2. Train/Test Splits & Variance Reduction Transformation
    # Converting to numpy vectors cleanly to align with the original notebook setup
    X_vals = X.values
    y_vals = np.log1p(y.values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_vals, y_vals, test_size=0.25, random_state=42
    )
    
    # 3. Isolation Forest Outlier Removal (Isolated to Training Set)
    X_train, y_train = clean_training_outliers(X_train, y_train)
    
    # 4. Train Models and Store Outputs
    models = get_baseline_models()
    results_output = []
    
    print("\n--- Model Evaluation Report ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        scores = evaluate_predictions(y_test, predictions)
        
        report_line = f"{name}:\n  R2  = {scores['R2']}\n  MAE = {scores['MAE']} years\n  MSE = {scores['MSE']}\n"
        print(report_line)
        results_output.append(report_line)
        
    # 5. Write execution results file
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.writelines("\n".join(results_output))
        
    print("Execution finalized. Metrics successfully logged inside results/results.txt")

if __name__ == "__main__":
    main()