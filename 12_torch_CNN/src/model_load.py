import torch
import pathlib
from .model import IntelCNN

PATH = pathlib.Path(__file__).resolve().parent.parent /'model'/"model.pth"

def save_model(model):
    torch.save(model.state_dict(), PATH)

def load_model():
    model = IntelCNN() 

    model.load_state_dict(torch.load(PATH))

    model.eval() 

    return model