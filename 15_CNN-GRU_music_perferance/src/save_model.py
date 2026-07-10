import logging
import torch
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def save_model(model, filename: str = "pann_gru.pth", model_dir: Path = MODEL_DIR):
    """Saves the model weights along with its Deep SVDD center coordinate vector."""
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / filename

    # Verify if the center has been computed
    center = getattr(model, "center", None)
    if center is None:
        logging.warning(
            "The model does not have a 'center' attribute computed. Saving weights only."
        )

    checkpoint = {"state_dict": model.state_dict(), "center": center}

    torch.save(checkpoint, save_path)
    logging.info(
        f"Successfully saved one-class model and center cluster to: {save_path}"
    )


def load_model(model, filename: str = "pann_gru.pth", model_dir: Path = MODEL_DIR):
    """Loads weights and assigns the deep center attribute back to the model instance."""
    load_path = model_dir / filename

    if not load_path.exists():
        raise FileNotFoundError(f"No trained model checkpoint found at: {load_path}")

    logging.info(f"Loading one-class checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location="cpu")

    # Restore the network weights
    model.load_state_dict(checkpoint["state_dict"])

    # Re-bind the one-class center vector back onto the object instance
    model.center = checkpoint.get("center", None)

    if model.center is None:
        logging.warning(
            "No 'center' vector was found in this checkpoint. Inference will require re-centering."
        )
    else:
        logging.info("One-class target cluster center restored successfully.")

    return model
