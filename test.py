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
from LSTMtorch import visualization, save_result, preprocessing

para = 'Res'
path = './dataset/A' + para + '.csv'
data_all = preprocessing(path)