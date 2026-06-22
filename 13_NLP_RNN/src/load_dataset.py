import torch
from torch.utils.data import Dataset
from .data_processing import lineToTensor, speak_merican
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'names'

class NamesDataset(Dataset):

    def __init__(self, data_dir = data_dir):
        self.data_dir = data_dir
        labels_set = set() #set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        for filepath in data_dir.glob('*.txt'):
            label = filepath.stem 
            labels_set.add(label)
            lines = filepath.read_text(encoding='utf-8').strip().split('\n')
            
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(speak_merican(name))) 
                self.labels.append(label)
        #Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item