import torch
import torch.nn as nn
# class CNN(torch.nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
#             torch.nn.BatchNorm2d(32),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.fc = torch.nn.Linear(50*32, 12)
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         out = self.conv(x)
#         out = out.view(out.size()[0], -1)
#         out = self.fc(out)
#         return out
# 定义Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# 定义ResNet模型
class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN, self).__init__()
        self.in_channels = 32  # 输入通道数，根据你的需求修改
        self.conv = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 根据你的需求修改
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(32, 2)
        self.layer2 = self.make_layer(64, 2, stride=2)
        self.layer3 = self.make_layer(128, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out