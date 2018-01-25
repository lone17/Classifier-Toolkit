# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

class svm_classifier(object):

    def __init__(self, X=None, y=None, data_file=None, header=0, test_size=0.2,
                 feature_col_range=[0, 8], label_col=-1, features_degree=1,
                 include_bias=True):

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size)

        self.poly = PolynomialFeatures(features_degree, include_bias=include_bias)
        self.X_train = self.poly.fit_transform(self.X_train)

        self.sc = StandardScaler()
        self.sc.fit_transform(self.X_train)

        self.model = None
        self.cm = None

    def fit(self, C=50, kernel='rbf', degree=3, gamma=10, coef0=0.0, max_iter=-1,
            strategy='ovr'):

        # Fitting Decision Tree Classification to the Training set
        print('Fitting the train set...')

        self.model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, max_iter=max_iter, decision_function_shape=strategy)

        X_train_norm = self.sc.transform(self.X_train)
        self.model.fit(X_train_norm, self.y_train)

        print('\nTrain Set Accuracy: ', self.model.score(X_train_norm, self.y_train) * 100)

        # Predicting the Test set results
        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            y_pred = self.predict(self.X_test)
            self.evaluate(X=self.X_test, y=self.y_test)

            # Making the Confusion Matrix
            self.cm = confusion_matrix(self.y_test, y_pred)

    def confusion_matrix(self):
        return self.cm

    def evaluate(self, X=None, y=None, data_file=None, header=0,
                 feature_col_range=[0, 8], label_col=-1):

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

        accuracy = self.model.score(X, y)
        print('Accuracy: ', accuracy * 100)

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

        return self.model.predict_proba(X)

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

        return self.model.predict(X)

if __name__ == '__main__':
    files = ["RD_1XST.csv", "RD_2P_P.csv", "RD_2X.csv", "RD_3P.csv", "RD_4P.csv",
             "RD_5P.csv", "RD_6P_P.csv", "RD_7P.csv", "RD_8P.csv", "RD-1P.csv",
             "RDT_1P.csv", "RDT_1RX.csv", "RDT_2P.csv"];

    # classifier = svm_classifier(data_file='RD-1P.csv')
    # classifier.fit()
    for file in files:
        classifier = svm_classifier(data_file=file)
        classifier.fit()
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
