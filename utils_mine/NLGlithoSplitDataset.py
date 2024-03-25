import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class NLGlithoSplitDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.label_file = label_file
        self.data_files = os.listdir(data_dir)
        self.label_mapping = self.load_label_mapping()

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        data = pd.read_csv(os.path.join(self.data_dir, file_name), header=None).values
        file_name_without_extension = os.path.splitext(file_name)[0]
        label = self.label_mapping.get(file_name_without_extension, 0)  # 0 is the default label if not found
        sample = {
            'data': torch.tensor(data, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
        return sample

    def load_label_mapping(self):
        label_mapping = {}
        if os.path.exists(self.label_file):
            label_data = pd.read_csv(self.label_file, header=None)
            for row in label_data.values:
                file_name_without_extension, label = str(row[0]), int(row[1])
                label_mapping[file_name_without_extension] = label
        return label_mapping
