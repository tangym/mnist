# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:26:24 2018

@author: tangym
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#%%
data = pd.read_csv('train.csv')        # 28*28
test = pd.read_csv('test.csv')


#%%
X = data[[col for col in data.columns if col.startswith('pixel')]].values
binarizer = LabelBinarizer()
y = binarizer.fit_transform(data['label'].values)
X_test = test[[col for col in data.columns if col.startswith('pixel')]].values
n_classes = len(data['label'].unique())

#%%
clf = Sequential()
clf.add(Reshape((28, 28, 1), input_shape=(784,)))
clf.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))
clf.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))
clf.add(Dropout(0.25))
clf.add(Flatten())
clf.add(Dense(units=128, activation='relu'))
clf.add(Dense(units=64, activation='relu'))
clf.add(Dense(n_classes, activation='softmax'))

clf.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
clf.fit(X, y)

clf.evaluate(X, y)

#%%
y_test = clf.predict(X_test)
y_labels = binarizer.inverse_transform(y_test)

pd.DataFrame({'ImageId': test.index + 1, 'Label': y_labels}).to_csv('result.csv', index=False)

