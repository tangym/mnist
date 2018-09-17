# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:58:39 2018

@author: tangym
"""

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

import pylab as plt


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
data = pd.read_csv('train.csv')        # 28*28
test = pd.read_csv('test.csv')

#%%
X = data[[col for col in data.columns if col.startswith('pixel')]].values
binarizer = LabelBinarizer()
y = binarizer.fit_transform(data['label'].values)
X_test = test[[col for col in data.columns if col.startswith('pixel')]].values
n_classes = len(data['label'].unique())

del data, test

#%%
model = C2F3(10)

criterion = nn.MSELoss(reduction='sum')
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-4)

#%%
n_samples = 800
tX = torch.tensor(X[:n_samples].reshape(n_samples, 1, 28, 28), dtype=torch.float)
ty = torch.tensor(y[:n_samples], dtype=torch.int)

#%%
for t in range(10):
    y_pred = model(tX)
    loss = criterion(y_pred, ty.float())
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#%%



#%%

