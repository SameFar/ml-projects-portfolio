# saves exclusively numpy one
import pickle
from pathlib import Path

model_path = Path(__file__).resolve().parent / 'model' / "fashion_nn_weights.pkl"

def save_model(model, filename = model_path):
    model_data = {
        "base_weights": model.weights,
        "base_bias": model.bias,
        "heads": model.heads
    }
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n---Brain successfully frozen---")
