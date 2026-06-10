import pickle
from pathlib import Path
from neural_network import NeuralNetwork

model_path = Path(__file__).resolve().parent / 'model' / "fashion_nn_weights.pkl"

def load_model():
    
    with open(model_path,'rb') as f:
        param = pickle.load(f)

    model = NeuralNetwork()
    model.weights = param['base_weights']
    model.bias = param['base_bias']
    model.heads = param['heads']

    return model