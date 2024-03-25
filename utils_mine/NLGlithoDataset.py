import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class NLGlithoDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'input': torch.tensor(self.X[idx], dtype=torch.float32),
            'target': torch.tensor(self.y[idx], dtype=torch.float32)
        }
        return sample
