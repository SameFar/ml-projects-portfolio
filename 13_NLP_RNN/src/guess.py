from .RNN import CharRNN
from .data_processing import lineToTensor, speak_merican
import torch

options = [
    "Italian",
    "Arabic",
    "Polish",
    "Dutch",
    "German",
    "Chinese",
    "Russian",
    "Spanish",
    "Greek",
    "Japanese",
    "English",
    "French",
    "Portuguese",
    "Irish",
    "Scottish",
    "Vietnamese",
    "Korean",
    "Czech",
]


def guess_name_origin(name, model_path):
    model = CharRNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X = lineToTensor(speak_merican(name))

    with torch.no_grad():
        pred = model.forward(X)
    _, guess_idx = pred.topk(1)

    return options[guess_idx]
