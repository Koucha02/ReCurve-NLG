import torch
import torch.nn as nn
class lithoLSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(lithoLSTM, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 1024)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(1024, 1024, batch_first=True, num_layers=1)
        self.fc4 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        # x = x.unsqueeze(1)  # 在第1维添加一个维度，形状变为（batch_size, 1, features）
        # print(x.shape)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc4(x)
        x = self.fc5(x)
        return x
