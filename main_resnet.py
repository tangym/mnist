# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 11:41:05 2018

@author: yut44
"""

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

import pylab as plt


#%% Try pretrained models
class CustomizedResNet(nn.Module):
    def __init__(self, n_classes=10):
        super(CustomizedResNet, self).__init__()
        self.upsample = nn.Upsample(size=(224, 224))
        self.resnet = models.resnet18(pretrained=True)
        # only tune the top layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        out = self.upsample(x)
        out = self.resnet(out.repeat(1, 3, 1, 1))
        return out 

#%%
model = CustomizedResNet(10)


#%%
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-4)

for t in range(10):
    y_pred = model(tX)
    loss = criterion(y_pred, ty.float())
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




#%% Try customize pretrained models
class CustomizedResNet(nn.Module):
    def __init__(self, n_classes=10):
        super(CustomizedResNet, self).__init__()
        self.upsample = nn.Upsample(size=(224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        #self.features = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(models.alexnet(pretrained=True).features.children()))
        self.fc = nn.Sequential(
                nn.Linear(1000, 128),
                nn.Linear(128, 64),
                nn.Linear(64, n_classes))

    def forward(self, x):
        out = self.upsample(x)
        out = self.normalize(out.repeat(1, 3, 1, 1))
        out = self.features(out)
        out = out.view(out.size(0), 256*6*6)
        out = self.fc(out)
        return out 


#%%
model = CustomizedResNet(10)

criterion = nn.MSELoss(reduction='sum')
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-4)

#%%
n_samples = 50
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
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                ])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


#%%