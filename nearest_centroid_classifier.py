# Nearest Centroid Classifier

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix

class near_centroid_classifier(object):

    def __init__(self, X=None, y=None, data_file=None, header=None, test_size=0.2,
                 feature_col_range=[0, 8], label_col=-1, features_degree=2,
                 include_bias=True):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        self.features_name = None

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            self.features_name = list(map(str, data))
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            y = data.iloc[:, label_col].values
        elif X is not None and y is not None:
            self.features_name = list(map(str, range(feature_col_range[1] - feature_col_range[0])))
            X = np.array(X)
            y = np.array(y)
        else:
            raise RuntimeError('Missing data')

        if include_bias:
            self.features_name.insert(0, 'Bias')


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size)

        self.poly = PolynomialFeatures(features_degree, include_bias=include_bias)
        self.X_train = self.poly.fit_transform(self.X_train)

        self.sc = StandardScaler()
        self.sc.fit(self.X_train)

        self.model = None
        self.cm = None

    def fit(self, metric='manhattan'):

        # Fitting Decision Tree Classification to the Training set
        print('Fitting the train set...')

        self.model = NearestCentroid(metric=metric, shrink_threshold=None)

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

    def plot(self, x_axis, y_axis, title=None):
        X_set, y_set = self.X_test, self.y_test
        X_set = self.poly.transform(X_set)
        X_set = self.sc.transform(X_set)
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, x_axis].min() - 1, stop = X_set[:, x_axis].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, y_axis].min() - 1, stop = X_set[:, y_axis].max() + 1, step = 0.01))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, x_axis], X_set[y_set == j, y_axis],
                        c = ListedColormap(('red', 'green', 'blue',
                                            'orange', 'purple', 'pink',
                                            'brown', 'gray', 'olive',
                                            'cyan'))(i), label = j)
        plt.title(title)
        plt.xlabel('Feature: ' + self.features_name[x_axis])
        plt.ylabel('Feature: ' + self.features_name[y_axis])
        plt.legend()
        plt.show()

if __name__ == '__main__':
    files = ["RD_1XST.csv", "RD_2P_P.csv", "RD_2X.csv", "RD_3P.csv", "RD_4P.csv",
             "RD_5P.csv", "RD_6P_P.csv", "RD_7P.csv", "RD_8P.csv", "RD-1P.csv",
             "RDT_1P.csv", "RDT_1RX.csv", "RDT_2P.csv"];

    # classifier = near_centroid_classifier(data_file='RD_1XST.csv')
    # classifier.fit()
    # classifier.plot(2,0)
    for file in files:
        classifier = near_centroid_classifier(data_file=file)
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
