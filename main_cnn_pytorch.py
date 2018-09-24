# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:58:39 2018

@author: tangym
"""

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader
from ignite.engine import Engine, create_supervised_trainer, Events
from ignite.handlers import Timer
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

import pylab as plt
from tqdm import tqdm

#%%
#dtype = torch.float
#device = torch.device('cpu')
#torch.set_default_dtype(dtype)

#%%
class C2F3(nn.Module):
    def __init__(self, n_classes=10):
        super(C2F3, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.layer2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.dropout = nn.Dropout2d()
        self.fc = nn.Sequential(
                nn.Linear(28*28*32, 128),
                nn.Linear(128, 64),
                nn.Linear(64, n_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out 

#%%
def batches(X, batch_size=1000):
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size]

#%% Prepare train/test data
data = pd.read_csv('train.csv')        # 28*28
test = pd.read_csv('test.csv')

X = data[[col for col in data.columns if col.startswith('pixel')]].values
binarizer = LabelBinarizer()
y = binarizer.fit_transform(data['label'].values)
X_test = test[[col for col in data.columns if col.startswith('pixel')]].values
n_classes = len(data['label'].unique())

X = torch.tensor(X.reshape(len(X), 1, 28, 28), dtype=torch.float)
y = torch.tensor(y, dtype=torch.int).float()
X_test = torch.tensor(X_test.reshape(len(X_test), 1, 28, 28), dtype=torch.float)

del data, test

#%%
class MNISTDataset(Dataset):
    def __init__(self, filename):
        data = pd.read_csv(filename)
        self.X = data[[col for col in data.columns if col.startswith('pixel')]].values
        self.y = LabelBinarizer().fit_transform(data['label'].values)
        self.X = torch.tensor(self.X.reshape(len(self.X), 1, 28, 28), dtype=torch.float)
        self.y = torch.tensor(self.y, dtype=torch.int).float()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

#%%
dataset = MNISTDataset('train_small.csv')


#%% Initialize model
model = C2F3(10)

criterion = nn.MSELoss(reduction='sum')
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-4)


#%%
for t in range(10):
    for batch_X, batch_y in zip(batches(X), batches(y)):
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        #print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#%%
engine = create_supervised_trainer(model, optimizer, criterion)

@engine.on(Events.ITERATION_COMPLETED)
def log_training_results(trainer):
    print(trainer.state.epoch, timer.value())



#%%
@engine.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

@engine.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

@engine.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

#%%
engine.run(zip(batches(X), batches(y)), max_epochs=3)

engine.run(DataLoader(dataset, batch_size=1000), max_epochs=2)


timer = Timer(average=True)
timer.attach(engine, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
             pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)


