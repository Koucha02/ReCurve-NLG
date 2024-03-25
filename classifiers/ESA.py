import torch
import torch.nn as nn
class ESALSTM(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, d_model=128, nhead=8, num_encoder_layers=4):
        super(ESALSTM, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # 用于将输入特征维度映射到模型维度
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers=num_encoder_layers
        )
        self.lstm = nn.LSTM(10, 128, batch_first=True, num_layers=3)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        output = self.embedding(x)  # 映射到模型维度
        output = output.unsqueeze(1)
        output = output.permute(1, 0, 2)  # 调整维度以适应Transformer的输入格式
        # print(x.shape)
        output = self.transformer_encoder(output)
        column_sums = output.sum(dim=2)
        # print(column_sums.shape)
        # 找到当前批量中最大k列的标号
        _, top_indices = torch.topk(column_sums, k=10, dim=1)
        # print(torch.sort(top_indices)[0])
        # output_fc = torch.index_select(output, dim=0, index=torch.sort(top_indices)[0])
        # 把这些列concat成一个新的tensor
        output = torch.gather(output, dim=2, index=torch.sort(top_indices)[0].unsqueeze(1).expand(-1, output.shape[1], -1))
        # print(output.shape)
        output, _ = self.lstm(output)
        output = output[-1, :, :]
        output = self.fc3(output)
        print(output.shape)
        return output

# class ESALSTM(nn.Module):
#     def __init__(self, input_dim=4, hidden_dim=128, n_layers=4, nhead=8, lstm_hidden_dim=128, output_dim=1, n_max_cols=2):
#         super(ESALSTM, self).__init__()
#         # Transformer Encoder
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead),
#             num_layers=n_layers
#         )
#         # LSTM layer
#         self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=1, batch_first=True)
#         # Output layer
#         self.fc = nn.Linear(lstm_hidden_dim, output_dim)
#         self.n_max_cols = n_max_cols
#
#     def forward(self, x):
#         # Transformer Encoder
#         attention_matrix = self.transformer_encoder(x)
#         # Sum along each column of the attention matrix
#         column_sums = attention_matrix.sum(dim=1)
#         # Find indices of top n_max_cols column sums
#         _, top_indices = torch.topk(column_sums, k=self.n_max_cols, dim=1)
#         # LSTM layer
#         lstm_out, _ = self.lstm(x)
#         # Select only the top n_max_cols columns from LSTM output
#         selected_lstm_out = torch.gather(lstm_out, dim=1, index=top_indices.unsqueeze(2).expand(-1, -1, lstm_out.size(2)))
#         # Output layer
#         output = self.fc(selected_lstm_out[:, :, -1])  # Assuming you want to use the last time step of LSTM output
#
#         return output

