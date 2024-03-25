import torch
import torch.nn as nn
class FC4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC4, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
