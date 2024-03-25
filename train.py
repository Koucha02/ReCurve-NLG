from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import tensorflow as tf
from keras.utils import plot_model
import csv


#转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        #预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


##数据预处理 load dataset
dataset = read_csv('./data_set/ZKN76-15.csv', header=0, index_col=0)
values = dataset.values
#标签编码 integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
#保证为float ensure all data is float
values = values.astype('float32')
#归一化 normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#转成有监督数据 frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#删除不预测的列 drop columns we don't want to predict
reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)
print(reframed.head())

#这里的reframe是把原数据数组复制了一遍之后删除掉不用的列，把剩下的行下移一个后并到原来的数组上，作为训练集

#数据准备
#把数据分为训练数据和测试数据 split into train and test sets
values = reframed.values
#拿一年的时间长度训练
n_train_hours = 4000
#划分训练数据和测试数据
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
#拆分输入输出 split into input and outputs
train_X, train_y = train[:, [0,1,2,3,4]], train[:, -1]
test_X, test_y = test[:, [0,1,2,3,4]], test[:, -1]
#reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

##模型定义 design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#模型训练 fit network
# train_X_processed = train_X[:, :, :36]
# test_X_processed = test_X[:, :, :36]
# history = model.fit(train_X_processed, train_y, epochs=8, batch_size=72, validation_data=(test_X_processed, test_y), verbose=2, shuffle=False)


history = model.fit(train_X, train_y, epochs=8, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
#输出 plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#进行预测 make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#yhat逆缩放 invert scaling for forecast
inv_yhat = concatenate((test_X[:, :-1], yhat), axis=1)
# print(inv_yhat[:, -1][1])
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]
inv_yhat = np.array(inv_yhat)

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, :-1], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

with open('Den_hat.csv', mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    for item in inv_yhat:
        writer.writerow([item])
with open('Den_true.csv', mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    for item in inv_y:
        writer.writerow([item])


# 画图
pyplot.plot(inv_y, label='true')
pyplot.plot(inv_yhat, label='prediction')
pyplot.legend()
pyplot.show()


# plot_model(model, show_shapes=True, to_file='model.png')
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
