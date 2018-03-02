# Importing the libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
np.set_printoptions(threshold=np.nan)

class Classifier:

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, feature_col_range=[1, 9], label_col=-1,
                 features_degree=1, features_scaling=True):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')
        self.col_range = feature_col_range
        self.label_col = label_col

        self.features_name = ['Bias', 'WELL', 'MD', 'TVDSS', 'GR', 'NPHI', 'RHOZ', 'DT',
                              'VCL', 'PHIE', 'Deltaic_Facies']

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, feature_col_range[0]:feature_col_range[1]]
            y = data[:, label_col].astype('int')
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size = test_size)
        elif train_set is not None and val_set is not None:
            train_set = np.array(train_set)
            val_set = np.array(val_set)
            self.X_train = train_set[:, feature_col_range[0]:feature_col_range[1]]
            self.X_test = val_set[:, feature_col_range[0]:feature_col_range[1]]
            self.y_train = train_set[:, label_col].astype('int')
            self.y_test = val_set[:, label_col].astype('int')
        else:
            raise RuntimeError('Missing data')

        self.poly = PolynomialFeatures(features_degree, include_bias=True)
        self.X_train = self.poly.fit_transform(self.X_train)

        self.features_scaling = features_scaling
        self.sc = StandardScaler()
        if features_scaling:
            self.X_train = self.sc.fit_transform(self.X_train)

        self.model = None
        self.cm = None
        self.score = 0

    def _post_process(self, y):
        for i in range(3, len(y)-3):
            if (y[i] != y[i-1]) or (y[i] != y[i+1]):
                if (y[i] != y[i+1]) and (y[i-1] == y[i-2] == y[i-3]):
                    y[i] = y[i-1]
                elif y[i+1] == y[i+2] == y[i+3]:
                    y[i] = y[i+1]

        return y

    def confusion_matrix(self):
        return self.cm.tolist()

    def evaluate(self, X=None, y=None, data_file=None, header=0):

        feature_col_range = self.col_range
        label_col = self.label_col

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            print('\nEvaluating on ', data_file, '...', sep='')
            X = data[:, feature_col_range[0]:feature_col_range[1]].values
            y = data[:, label_col]
        elif X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        if self.features_scaling:
            X = self.sc.transform(X)

        prob = self.model.predict_proba(X)
        loss = log_loss(y, prob)

        pred = prob.argmax(axis=1)
        accuracy = np.count_nonzero(y == pred) / len(y) * 100
        print('Accuracy: ', accuracy, ' Loss: ', loss)

        pred = self._post_process(pred)
        accuracy = np.count_nonzero(y == pred) / len(y) * 100
        print('Accuracy after post-processing: ', accuracy)

        self.cm = confusion_matrix(y, pred, labels=[i for i in range(11)])

        return [loss, accuracy]

    def probality(self, X=None, data_file=None, header=0):

        feature_col_range = self.col_range

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data[:, feature_col_range[0]:feature_col_range[1]].values
        elif X is not None:
            X = np.array(X)
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        if self.features_scaling:
            X = self.sc.transform(X)

        return self.model.predict_proba(X).tolist()

    def predict(self, X=None, data_file=None, header=0):

        feature_col_range = self.col_range

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data[:, feature_col_range[0]:feature_col_range[1]].values
        elif X is not None:
            X = np.array(X)
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        if self.features_scaling:
            X = self.sc.transform(X)

        return self._post_process(self.model.predict(X)).tolist()

    def plot(self, x_axis, y_axis, title=None):
        X_set, y_set = self.X_test, self.y_test
        # X_set = StandardScaler().fit_transform(X_set)
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, x_axis].min() - 1,
                                       stop=X_set[:, x_axis].max() + 1, step=0.01),
                             np.arange(start=X_set[:, y_axis].min() - 1,
                                       stop=X_set[:, y_axis].max() + 1, step=0.01))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, x_axis], X_set[y_set == j, y_axis],
                        c = ListedColormap(('red', 'green', 'blue',
                                            'orange', 'purple', 'pink',
                                            'brown', 'gray', 'olive',
                                            'cyan', '#59ec45'))(i), label = j)
        plt.title(title)
        plt.xlabel('Feature: ' + self.features_name[x_axis])
        plt.ylabel('Feature: ' + self.features_name[y_axis])
        plt.legend()
        plt.show()


    def load_model(self, file_name):
        self.model = joblib.load(file_name)

def merge_data(files, header=0):
    data = pd.DataFrame()
    for file in files:
        data =  pd.concat([data, pd.read_csv(file, header=header)])

    return data.values

files = ["RD_1XST.csv", "RD_2P_P.csv", "RD_2X.csv", "RD_3P.csv", "RD_4P.csv",
         "RD_5P.csv", "RD_6P_P.csv", "RD_7P.csv", "RD_8P.csv", "RD-1P.csv",
         "RDT_1P.csv", "RDT_1RX.csv", "RDT_2P.csv", "RD-RDT DATA ALL.csv"];

group1 = ['RD_1XST.csv', 'RD_2X.csv', 'RD_3P.csv', 'RD_4P.csv']
group2 = ['RDT_2P.csv', 'RD_2P_P.csv', 'RD_7P.csv', 'RDT_1P.csv']
group3 = ['RD-1P.csv', 'RD_5P.csv', 'RD_6P_P.csv', 'RD_8P.csv']

np.random.seed(17)
