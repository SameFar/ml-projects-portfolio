import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

file_path = Path(__file__).resolve().parent.parent / 'nvidea_stocks2.csv'

data = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=(1, 2, 3, 4), dtype=float)

def train_test_split(data=data, train_size=0.7, timestep=50):
    split_line = int(len(data) * train_size)
    train_data = data[:split_line]

    start_test_idx = max(0, split_line - timestep)
    test_data = data[start_test_idx:]
    
    return train_data, test_data

def make_loader(data, timestep=50):
    X, y = [], []

    for i in range(len(data) - timestep):
        X.append(data[i : i + timestep, 0:3])

        y.append(data[i + timestep - 1, 3])

    Xt = torch.tensor(np.array(X), dtype=torch.float32)
    yt = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=32, shuffle=False)

def reverse_scale_pred(pred, min_val, max_val): 
    return (pred * (max_val - min_val)) + min_val