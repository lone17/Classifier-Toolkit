import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

class neural_network_classifier(object):

    def __init__(self, X=None, y=None, data_file=None, header=0, num_labels=None,
                 test_size=0.2, feature_col_range=[0, 8], label_col=-1,
                 features_degree=2, include_bias=True):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            y = data.iloc[:, label_col].values
        elif X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
        else:
            raise RuntimeError('Missing data')

        self.his = None
        self.cm = None

        if num_labels is None:
            self.num_labels = max(y) + 1
        else:
            self.num_labels = num_labels

        if not (self.num_labels*1.0).is_integer:
            raise RuntimeError('Invalid labels')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size)

        self.poly = PolynomialFeatures(features_degree, include_bias=include_bias)
        self.X_train = self.poly.fit_transform(self.X_train)

        _, self.num_features = self.X_train.shape

        self.sc = StandardScaler()
        self.sc.fit(self.X_train)

        self.model = None

        self.structure()

    def structure(self,
                  hidden_layers=[{'activate': 'sigmoid', 'units': 30}],
                  output_layer_activation='sigmoid'):

        self.model = Sequential()

        self.model.add(Dense(activation=hidden_layers[0]['activate'],
                             input_dim=self.num_features,
                             units=hidden_layers[0]['units'],
                             kernel_initializer='glorot_uniform'))

        for layer in hidden_layers[1:]:
            self.model.add(Dense(activation=layer['activate'],
                                 units=layer['units'],
                                 kernel_initializer='uniform'))

        self.model.add(Dense(activation=output_layer_activation,
                             units=self.num_labels,
                             kernel_initializer='uniform'))

    def train(self, batch_size=1, num_epochs=150, optimizer='sgd', learning_rate=None):

        if learning_rate is not None:
            if optimizer == 'sgd':
                optimizer = keras.optimizers.SGD(lr=learning_rate)
            elif optimizer == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(lr=learning_rate)
            elif optimizer == 'adagrad':
                optimizer = keras.optimizers.Adagrad(lr=learning_rate)
            elif optimizer == 'adadelta':
                optimizer = keras.optimizers.Adadelta(lr=learning_rate)
            elif optimizer == 'adam':
                optimizer = keras.optimizers.Adam(lr=learning_rate)
            elif optimizer == 'adamax':
                optimizer = keras.optimizers.Adamax(lr=learning_rate)
            elif optimizer == 'nadam':
                optimizer = keras.optimizers.Nadam(lr=learning_rate)

        print('\nTraining Neural Network...\n')

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.his = self.model.fit(self.sc.transform(self.X_train),
                                  to_categorical(self.y_train, self.num_labels),
                                  batch_size=batch_size,
                                  epochs=num_epochs,
                                  verbose=1)

        train_accuracy = self.his.history['acc'][-1]
        train_loss = self.his.history['loss'][-1]

        print('\nTrain Set Accuracy: ', train_accuracy * 100, '  Loss: ', train_loss)

        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            X = self.poly.transform(self.X_test)
            X = self.sc.transform(X)
            y_pred = self.model.predict(X).argmax(axis=1)
            self.cm = confusion_matrix(self.y_test, y_pred)
            self.evaluate(X=self.X_test, y=self.y_test)

    def confusion_matrix(self):
        return self.cm

    def evaluate(self, X=None, y=None, data_file=None, header=0,
                 batch_size=1, feature_col_range=[0, 8], label_col=-1):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            print('\nEvaluating on ', data_file, '...', sep='')
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            y = data.iloc[:, label_col]
        elif X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        X = self.sc.transform(X)
        y = to_categorical(y, self.num_labels)

        [loss, accuracy] = self.model.evaluate(X, y, batch_size=batch_size)
        print('Accuracy: ', accuracy * 100, '  Loss: ', loss)

    def probality(self, X=None, data_file=None, header=0, feature_col_range=[0, 8]):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
        elif X is not None and y is not None:
            X = np.array(X)
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        X = self.sc.transform(X)

        return self.model.predict(X, verbose=1)

    def predict(self, X=None, data_file=None, header=0, feature_col_range=[0, 8]):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
        elif X is not None:
            X = np.array(X)
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        X = self.sc.transform(X)
        prob = self.model.predict(X, verbose=1)

        return prob.argmax(axis=1)

    def plot(self, name='loss and accuracy per epoch'):
        plt.plot(self.his.history['loss'])
        plt.plot(self.his.history['acc'])
        plt.title(name)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

if __name__ == '__main__':
    files = ["RD_1XST.csv", "RD_2P_P.csv", "RD_2X.csv", "RD_3P.csv", "RD_4P.csv",
             "RD_5P.csv", "RD_6P_P.csv", "RD_7P.csv", "RD_8P.csv", "RD-1P.csv",
             "RDT_1P.csv", "RDT_1RX.csv", "RDT_2P.csv"];

    # for file in files:
    #     classifier = neural_network_classifier(data_file=file, num_labels=11)
    #     classifier.structure([{'activate': 'sigmoid', 'units': 30}],
    #                          output_layer_activation='sigmoid')
    #     classifier.train(num_epochs=150, batch_size=1, optimizer='sgd')
        # print(classifier.confusion_matrix())
        # classifier.evaluate(data_file='RD_2P_P.csv')
        # classifier.evaluate(data_file='RD_2X.csv')
        # classifier.evaluate(data_file='RD_3P.csv')
        # classifier.evaluate(data_file='RD_4P.csv')
        # classifier.evaluate(data_file='RD_5P.csv')
        # classifier.evaluate(data_file='RD_6P_P.csv')
        # classifier.evaluate(data_file='RD_7P.csv')
        # classifier.evaluate(data_file='RD_8P.csv')
        # classifier.evaluate(data_file='RD-1P.csv')
        # classifier.evaluate(data_file='RDT_1P.csv')
        # classifier.evaluate(data_file='RDT_1RX.csv')
        # classifier.evaluate(data_file='RDT_2P.csv')

    classifier = neural_network_classifier(data_file='RD_1XST.csv', num_labels=11)
    classifier.structure([{'activate': 'sigmoid', 'units': 30}],
    output_layer_activation='sigmoid')
    classifier.train(num_epochs=200, batch_size=1, optimizer='adam')
    classifier.plot()
