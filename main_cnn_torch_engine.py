# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:39:53 2018

@author: yut44
"""
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.handlers import Timer

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

import pylab as plt
from tqdm import tqdm


#%% Define the model
class C2F3(nn.Module):
    def __init__(self, n_classes=10):
        super(C2F3, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.dropout = nn.Dropout2d()
        self.fc = nn.Sequential(
                nn.Linear(28*28*32, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, n_classes), nn.Softmax(dim=0))
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

#%% Define the dataset
class MNISTDataset(Dataset):
    def __init__(self, filename):
        data = pd.read_csv(filename)
        self.X = data[[col for col in data.columns if col.startswith('pixel')]].values
        # Convert label to one-hot encoding
        self.y = np.zeros((len(data), len(data['label'].unique())))
        self.y[np.arange(len(data)), data['label'].values] = 1
        
        # Convert X and y to torch tensor
        self.X = torch.tensor(self.X.reshape(len(self.X), 1, 28, 28), dtype=torch.float)
        self.y = torch.tensor(self.y, dtype=torch.float)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
    
    def __len__(self):
        return len(self.X)

#%%
model = C2F3(n_classes=10)
dataset = MNISTDataset('train_small.csv')
train_loader = DataLoader(dataset, batch_size=1024)
#eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

criterion = nn.MSELoss(reduction='sum')    #nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-4)    #torch.optim.SGD(model.parameters(), lr=1e-4)

#%% Initialize trainer and handlers which print info after iterations and epochs
trainer = create_supervised_trainer(model, optimizer, criterion)
evaluator = create_supervised_evaluator(model, 
                                        metrics={
                                                #'accuracy': CategoricalAccuracy(),
                                                'mse': Loss(criterion)
                                        })
timer = Timer(average=True)
timer.attach(trainer, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED,
             resume=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)
    
@trainer.on(Events.ITERATION_COMPLETED)
def log_on_iteration_completed(trainer):
    print('Epoch [{}] Loss: {:.2f}'.format(
            trainer.state.epoch, trainer.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_on_epoch_completed(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    #print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #      .format(trainer.state.epoch, metrics['accuracy'], metrics['mse']))
    print('Epoch [{}] Time: {:.2f} sec'.format(trainer.state.epoch, timer.value()))
    print('Training set - Epoch {}: Loss {:.2f}'
          .format(trainer.state.epoch, metrics['mse']))#metrics['accuracy']))#, metrics['mse']))

#%%
trainer.run(train_loader, max_epochs=2)

#%%

