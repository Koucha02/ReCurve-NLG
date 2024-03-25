import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import csv
import os
from math import sqrt
import torch.nn.functional as F
import random

# 转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 拼接 put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def preprocessing(path):
    global scaler
    global scaled
    global reframed
    dataset = pd.read_csv(path, header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)
    # 删除不预测的列
    reframed.drop(reframed.columns[[4,5,6,7,8]], axis=1, inplace=True)
    # 这里的reframe是把原数据数组复制了一遍之后删除掉不用的列，把剩下的行下移一个后并到原来的数组上，作为训练集
    # 把数据分为训练数据和测试数据 split into train and test sets
    values = reframed.values
    # 拿2/3长度训练
    n_train_len = 0
    # 划分训练数据和测试数据
    train = values[:n_train_len, :]
    test = values[n_train_len:, :]
    # 拆分输入输出 split into input and outputs
    train_X, train_y = train[:, [0, 1, 2, 3, 4]], train[:, -1]
    test_X, test_y = test[:, [0, 1, 2, 3, 4]], test[:, -1]
    # reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


def visualize_heatmap(tensor):
    squeezed_tensor = tensor.squeeze(0)# 压缩第一维度
    heatmap = squeezed_tensor.detach().numpy()  # 转换为NumPy数组
    # 绘制热力图
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.xlabel('Batch Index')  # 添加X轴标签
    plt.ylabel('Batch Index')  # 添加Y轴标签
    plt.title('ESA-LSTM Attention Matrix')  # 添加图片标题
    plt.colorbar()  # 添加颜色条
    save_path = './heatmap' + '.png'  # 保存路径和文件名
    plt.savefig(save_path)  # 保存热力图
    # plt.show()

# 示例使用

def get_dominant_index(input_tensor, k):
    sequence_length, _, input_dim = input_tensor.size()
    multihead_attention = nn.MultiheadAttention(embed_dim=5, num_heads=5)
    _, attention_weights = multihead_attention(input_tensor, input_tensor, input_tensor)
    # visualize_heatmap()
    mean_attention_weights = attention_weights.mean(dim=1)
    # 找值最大的k列
    dominant_columns = torch.topk(mean_attention_weights, k=k, dim=-1).indices
    return dominant_columns

#全连接层版本
class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(5, 25)
        self.bn1 = nn.BatchNorm1d(1)  # 修改批量归一化层的维度
        self.dropout = nn.Dropout(0.8)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 基础LSTM版本
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.fc1 = nn.Linear(5, 25)
        self.lstm = nn.LSTM(input_size=25, hidden_size=25, num_layers=1)
        self.bn1 = nn.BatchNorm1d(1)  # 修改批量归一化层的维度
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lstm(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x[:, -1, :])  # 取最后一个时间步的输出
        return x

class ESALSTM(nn.Module):
    def __init__(self):
        super(ESALSTM, self).__init__()
        self.fc1 = nn.Linear(5, 30)
        self.lstm = nn.LSTM(input_size=30, hidden_size=30, num_layers=1)
        self.bn1 = nn.BatchNorm1d(1)  # 修改批量归一化层的维度
        self.fc2 = nn.Linear(30, 1)

    def forward(self, x):
        indices = get_dominant_index(x, int(0.3*x.shape[0]))
        indices, _ = torch.sort(indices)
        indices = indices.squeeze()
        # print("xx", indices)
        x = self.fc1(x)
        all_indices = torch.arange(x.shape[0])
        remaining_indices = torch.tensor(list(set(all_indices.tolist()) - set(indices.tolist())))
        # print(remaining_indices)
        # 选取标号以外的列数据
        x_fc = torch.index_select(x, dim=0, index=remaining_indices).squeeze(1)
        x_lstm = torch.index_select(x, dim=0, index=indices.long())
        # print(x_lstm.shape)
        if x_lstm.shape[0] > 0:  # 只有在x_lstm的序列长度大于0时才进行LSTM计算
            x_lstm, _ = self.lstm(x_lstm)
            x_lstm = x_lstm[:, -1, :]  # 取最后一个时间步的输出
        else:
            x_lstm = torch.zeros(0, self.lstm.hidden_size).to(x.device)  # 若x_lstm序列长度为0，则创建一个空的张量
        combined_tensor = torch.zeros((x_fc.shape[0] + x_lstm.shape[0], x_fc.shape[1]))
        combined_tensor[remaining_indices.long(), :] = x_fc
        # 将第二个tensor的数据放入合并后的tensor中
        combined_tensor[indices.long(), :] = x_lstm
        x = combined_tensor
        x = self.fc2(x)
        return x

def model_def(data_all, mode):
    global loss
    global losses

    if model_type == 0:
        model = FCModel()
    if model_type == 1:
        model = LSTMModel()
    if model_type == 2:
        model = ESALSTM()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    train_X, train_y, test_X, test_y = data_all
    train_dataset = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_y))
    test_dataset = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100)

    if mode == 0:
        # 模型训练 fit network
        losses = []
        for epoch in range(train_epoch):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs.squeeze(), targets).item()
            losses.append(loss.item())
            print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, train_epoch, loss.item(), val_loss))
    else:
        if model_type == 0:
            model = FCModel()
        if model_type == 1:
            model = LSTMModel()
        if model_type == 2:
            model = ESALSTM()
        model.load_state_dict(torch.load(model_path))
        model.eval()

    # 进行预测 make a prediction
    model.eval()
    yhat = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            batch_outputs = model(inputs)
            # yhat.append(outputs.item())
            yhat.extend(batch_outputs.squeeze().tolist())
    inv_yhat = np.array(yhat)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = np.column_stack((test_X[:, :-1], inv_yhat))
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    inv_y = scaler.inverse_transform(test_X)
    inv_y = inv_y[:, -1]

    if mode == 0:
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training history')
        plt.show()
        losses = np.array(losses)
        print(type(losses))
        torch.save(model.state_dict(), './models/fc-cal')

    return inv_y, inv_yhat

def visualization(ori, pred):
    # 画图
    plt.plot(ori, label='true')
    plt.plot(pred, label='prediction')
    plt.legend()
    plt.show()
    # calculate RMSE
    rmse = sqrt(mean_squared_error(ori, pred))

    print('Test RMSE: %.3f' % rmse)

def save_result(ori, pred):
    # 保存预测结果和真是值
    with open(para+'_hat.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for item in pred:
            writer.writerow([item])
    with open(para+'_true.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for item in ori:
            writer.writerow([item])

    # 保存loss，以便后续画图
    filename = 'loss_100.csv'
    # 检查文件是否存在
    file_exists = os.path.isfile(filename)
    # 以追加模式打开文件，并选择适当的写入模式
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # 如果文件为空，则直接写入变量数据
        if not file_exists:
            writer.writerow(losses)
        else:
            # 如果文件非空，则在新行写入变量数据
            writer.writerow(losses)

if __name__ == '__main__':
    """Res/Den/Ng/Cal/SP"""
    para = 'Res'
    model_type = 2
    #选择模型，0为全连接层，1为LSTM，2为ESA-LSTM
    train_epoch = 9
    path = './dataset/E' + para + '.csv'
    # path = './dataset/ZKN76-15' + para + '.csv'
    data_all = preprocessing(path)

    #模型定义 0时从头训练，1时加载模型
    model_path = './models/elstm-res'
    inv_y, inv_yhat = model_def(data_all, mode=0)
    visualization(inv_y, inv_yhat)
    save_result(inv_y, inv_yhat)

    