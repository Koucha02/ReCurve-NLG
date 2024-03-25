import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from utils_mine.NLGlithoDataset import NLGlithoDataset
from utils_mine.NLGseq2seqDataset import NLGseq2seqDataset
from classifiers.lithoLSTM import lithoLSTM
from classifiers.FC4 import FC4
from classifiers.FC8 import FC8
from classifiers.ESA import ESALSTM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

input_dim = 4
output_dim = 1

model = lithoLSTM(input_dim, output_dim)
model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

csv_file = './dataset/ZKN76-15Res.csv'
dataset = NLGseq2seqDataset(csv_file, 100)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch in train_loader:

        inputs, labels = batch['input'], batch['target']
        # print(inputs.shape)
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs)
        # print(outputs.shape())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def test(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['input'], batch['target']

            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return test_loss / len(test_loader), accuracy

num_epochs = 100

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss, accuracy = test(model, test_loader, criterion)
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

# torch.save(model, "models/model-fc-f.pkl")
#inference