import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix

import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from scipy.optimize import minimize

data = pd.read_csv('RD-1P.csv', header=0)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, )

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = to_categorical(y_train, 11)
y_test = to_categorical(y_test, 11)

num_samples, _ = X_train.shape

model = Sequential()

model.add(Dense(activation='sigmoid', input_dim=8, units=15, use_bias=False))
model.add(Dense(activation='sigmoid', units=30, use_bias=False))
model.add(Dense(activation='sigmoid', units=11, use_bias=False))
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
# his = model.fit(X_train, y_train, batch_size=1000, epochs=100)

def fold(w):
    weights = []
    for layer in model.layers:
        shape = layer.get_weights()[0].shape
        elements = shape[0] * shape[1]
        weights.append(np.reshape(w[:elements], shape))
        w = w[elements:]
    return weights

def result(w):
    weights = fold(w)
    for w, layer in zip(weights, model.layers):
        layer.set_weights([w])

    res = model.evaluate(X_train, y_train, batch_size=num_samples, verbose=0)

    return res

num_w = 0
w = []
for layer in model.layers:
    cur_w = layer.get_weights()[0]
    shape = cur_w.shape
    num_w += shape[0] * shape[1]
    w = np.concatenate((w, np.ravel(cur_w)))

npop = 100
sigma = 0.01
alpha = 0.001
epochs = 1000

# r = his.history['loss'][-1]
r = 10
for i in range(epochs):
    N = np.random.randn(npop, num_w)
    R = np.zeros(npop)
    # new_w = w.copy()
    for j in range(npop):
        w_try = w + sigma * N[j]
        res = result(w_try)[0]
        R[j] = -res
    #     if res <= r:
    #         print(res)
    #         r = res
    #         new_w = w_try
    # w = new_w
    A = (R - np.mean(R)) / np.std(R)
    w = w + alpha / (npop*sigma) * np.dot(N.T, A)

    [loss, acc] = result(w)
    print('epoch %d/%d. loss: %f, accuracy: %f' % (i+1, epochs, loss, acc*100))
    r = acc

# r = his.history['loss'][-1]
# r = 10
for i in range(100):
    N = np.random.randn(npop, num_w)
    R = np.zeros(npop)
    new_w = w.copy()
    for j in range(npop):
        w_try = w + sigma * N[j]
        res = result(w_try)[1]
        # R[j] = -res
        if res > r:
            print(res)
            r = res
            new_w = w_try
    w = new_w
    # A = (R - np.mean(R)) / np.std(R)
    # w = w + alpha / (npop*sigma) * np.dot(N.T, A)

    [loss, acc] = result(w)
    print('epoch %d/%d. loss: %f, accuracy: %f' % (i+1, 100, loss, acc*100))

test_result = model.evaluate(X_test, y_test, batch_size=1, verbose=0)
print(test_result)
