import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import d2l

# class lithoTransformer(nn.Module):
#     def __init__(self, input_dim=10, n_classes=12, d_model=128, nhead=8, num_encoder_layers=4):
#         super(lithoTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, d_model)  # 用于将输入特征维度映射到模型维度
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead),
#             num_layers=num_encoder_layers
#         )
#         self.fc3 = nn.Linear(d_model, n_classes)
#
#     def forward(self, x):
#         x = self.embedding(x)  # 映射到模型维度
#         x = x.permute(1, 0, 2)  # 调整维度以适应Transformer的输入格式
#         output = self.transformer_encoder(x)
#         # 取序列的最后一行作为输出
#         output = output[-1, :, :]
#         output = self.fc3(output)
#         return output
class lithoTransformer(nn.Module):
    def __init__(self, input_dim=200, n_classes=20, d_model=128, nhead=8, num_encoder_layers=4):
        super(lithoTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # 用于将输入特征维度映射到模型维度
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers=num_encoder_layers
        )
        self.fc3 = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)  # 映射到模型维度
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)  # 调整维度以适应Transformer的输入格式
        # print(x.shape)
        output = self.transformer_encoder(x)
        # 取序列的最后一行作为输出
        output = output[-1, :, :]
        output = self.fc3(output)
        return output
