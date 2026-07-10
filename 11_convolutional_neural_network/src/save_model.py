import pickle
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent / "model" / "CNN_weights.pkl"


def save_model(model, filename=model_path):
    model_data = {
        "base_depth": model.depth,
        "in_depth": model.in_depth,
        "base_bias": model.conv_biases,
        "kernals": model.kernals,
        "dense_weights": model.dense_weights,
        "dense_bias": model.dense_biases,
    }
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)

    print("\n---Brain successfully frozen---")


def load_model(filename=model_path, img_shape=[28, 28]):
    from cnn import CompleteCNN

    with open(filename, "rb") as f:
        param = pickle.load(f)

    input_size = (param["in_depth"], img_shape[0], img_shape[1])

    model = CompleteCNN(depth=param["base_depth"], input_size=input_size)

    model.conv_biases = param["base_bias"]
    model.kernals = param["kernals"]
    model.dense_weights = param["dense_weights"]
    model.dense_biases = param["dense_bias"]

    print(f"---Brain successfully unfrozen from {filename.name}---")
    return model
