import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class NLGseq2seqDataset(Dataset):
    def __init__(self, csv_file, num_rows):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        self.num_rows = num_rows

    def __len__(self):
        return len(self.data) // self.num_rows

    def __getitem__(self, idx):
        start_idx = idx * self.num_rows
        end_idx = (idx + 1) * self.num_rows

        sample = {
            'input': torch.tensor(self.X[start_idx:end_idx], dtype=torch.float32),
            'target': torch.tensor(self.y[start_idx:end_idx], dtype=torch.float32)
        }
        return sample