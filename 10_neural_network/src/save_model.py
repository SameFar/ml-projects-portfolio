# saves exclusively numpy one
import pickle
from pathlib import Path

model_path = Path(__file__).resolve().parent / "model" / "fashion_nn_weights.pkl"
production_path = (
    Path(__file__).resolve().parent.parent / "api" / "model" / "fashion_nn_weights.pkl"
)


def save_model(model, filename=model_path, pfilename=production_path, production=True):
    model_data = {
        "base_weights": model.weights,
        "base_bias": model.bias,
        "heads": model.heads,
    }
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)

    if production:
        with open(pfilename, "wb") as f:
            pickle.dump(model_data, f)

    print("\n---Brain successfully frozen---")
