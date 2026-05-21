from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from data_loader import load_and_engineer_data
from model import get_baseline_pipeline

def main():
    # Structural relative file tracks
    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Fetch traing/testing splits
    print("Loading datasets and running pipeline transforms...")
    X_train, X_test, y_train, y_test = load_and_engineer_data()

    # 2. Train model
    print("Fitting Baseline XGBoost Classifier Pipeline...")
    pipe = get_baseline_pipeline()
    pipe.fit(X_train, y_train)
    
    # 3. Predict | metrics output
    preds = pipe.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    # 4. Write performance sheet
    with open(RESULTS_DIR / "results.txt", "w") as f:
        f.write("=== Credit Card Fraud Baseline Evaluation ===\n")
        f.write(f"Evaluated Test Observations: {len(y_test)}\n")
        f.write(f"Precision Score: {precision:.4f}\n")
        f.write(f"Recall Score: {recall:.4f}\n\n")
        f.write("Confusion Matrix Array Breakdown:\n")
        f.write(f"True Negatives (Legit caught): {cm[0][0]}\n")
        f.write(f"False Positives (Type I error): {cm[0][1]}\n")
        f.write(f"False Negatives (Type II error - Missed Fraud): {cm[1][0]}\n")
        f.write(f"True Positives (Fraud caught): {cm[1][1]}\n")

    print(f"Log sheet completed. Baseline Recall: {recall:.4f}")

    # 5. Visual Matrix Export
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=["Not Fraud", "Fraud"], 
                yticklabels=["Not Fraud", "Fraud"])
    plt.ylabel("Actual State")
    plt.xlabel("Predicted State")
    plt.title("Fraud Evaluation Confusion Matrix")
    plt.tight_layout()
    
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()
    print("Confusion Matrix plot exported successfully.")

if __name__ == "__main__":
    main()