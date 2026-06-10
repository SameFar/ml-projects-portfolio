import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from data_loader import load_clean_split_data
from model import get_regression_pipeline

def main():
    # Dynamic path handling relative to this script's position
    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 1. Ingest split matrices
    print("Fetching and filtering California Housing data...")
    X_train, X_test, y_train, y_test = load_clean_split_data()
    
    # 2. Initialize and compile training pipeline
    print("Training HistGradientBoostingRegressor Pipeline...")
    pipeline = get_regression_pipeline()
    pipeline.fit(X_train, y_train)
    
    # 3. Generate predictions
    predictions = pipeline.predict(X_test)
    
    # 4. Compute performance metrics
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # 5. Log metrics summary
    with open(RESULTS_DIR / "results.txt", "w") as f:
        f.write("=== California Housing Pipeline Run Results ===\n")
        f.write(f"Train Shape: {X_train.shape}\n")
        f.write(f"Test Shape: {X_test.shape}\n")
        f.write(f"Testing R2 Score: {r2:.4f}\n")
        f.write(f"Testing RMSE (Log scale): {rmse:.4f}\n")
        
    print(f"Metrics saved. R2 Score: {r2:.4f}")
    
    # 6. Generate and export assessment visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.4, color="teal")
    
    # Draw reference diagonal line for perfect predictions
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--r', lw=2, label="Perfect Alignment")
    
    plt.xlabel("True Values (Log Scale)")
    plt.ylabel("Predictions (Log Scale)")
    plt.title("California Housing: Predictions vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(RESULTS_DIR / "prediction_analysis.png")
    plt.close()
    print("Evaluation visualization exported successfully.")

if __name__ == "__main__":
    main()