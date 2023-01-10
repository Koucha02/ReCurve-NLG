import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)


