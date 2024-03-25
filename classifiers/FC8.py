import torch
import torch.nn as nn
class FC8(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC8, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(2048, 4096)
        self.relu = nn.ReLU()
        self.fc7 = nn.Linear(4096, 8192)
        self.relu = nn.ReLU()
        self.fc8 = nn.Linear(8192, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        return x
