# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.externals import joblib

class naive_bayes_classifier(object):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0, test_size=0.2,
                 feature_col_range=[1, 8], label_col=-1, features_degree=1,
                 include_bias=True, features_scaling=False):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            y = data.iloc[:, label_col].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size)
            self.features_name = list(map(str, range(feature_col_range[1] - feature_col_range[0])))
        elif train_set is not None and val_set is not None:
            self.X_train = train_set.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            self.X_test = val_set.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            self.y_train = train_set.iloc[:, label_col].values
            self.y_test = val_set.iloc[:, label_col].values
            self.features_name = list(map(str, range(feature_col_range[1] - feature_col_range[0])))
        else:
            raise RuntimeError('Missing data')

        if include_bias:
            self.features_name.insert(0, 'Bias')

        self.poly = PolynomialFeatures(features_degree, include_bias=include_bias)
        self.X_train = self.poly.fit_transform(self.X_train)

        self.features_scaling = features_scaling
        self.sc = StandardScaler()
        if features_scaling:
            self.X_train = self.sc.fit_transform(self.X_train)

        self.model = None
        self.cm = None
        self.score = 0

    def fit(self, strategy=None):

        estimator = BernoulliNB()

        if strategy is None:
            self.model = estimator
        elif strategy == 'ovo':
            self.model = OneVsOneClassifier(estimator=estimator)
        elif strategy == 'ovr':
            self.model = OneVsRestClassifier(estimator=estimator)
        else:
            raise RuntimeError('strategy can only be \'ovo\' or \'ovr\' or None')

        # Fitting Decision Tree Classification to the Training set
        print('Fitting the train set...')

        self.model.fit(self.X_train, self.y_train)

        print('\nTrain Set Accuracy: ', self.model.score(self.X_train, self.y_train) * 100)

        # Predicting the Test set results
        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            y_pred = self.predict(self.X_test)
            self.score = self.evaluate(X=self.X_test, y=self.y_test)

            # Making the Confusion Matrix
            self.cm = confusion_matrix(self.y_test, y_pred)

    def confusion_matrix(self):
        return self.cm

    def evaluate(self, X=None, y=None, data_file=None, header=0,
                 feature_col_range=[1, 8], label_col=-1):

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
        if self.features_scaling:
            X = self.sc.transform(X)

        accuracy = self.model.score(X, y)
        print('Accuracy: ', accuracy * 100)

        return accuracy

    def probality(self, X=None, data_file=None, header=0, feature_col_range=[1, 8]):

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
        if self.features_scaling:
            X = self.sc.transform(X)

        return self.model.predict_proba(X)

    def predict(self, X=None, data_file=None, header=0, feature_col_range=[1, 8]):

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
        if self.features_scaling:
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
                                            'cyan', '#59ec45'))(i), label = j)
        plt.title(title)
        plt.xlabel('Feature: ' + self.features_name[x_axis])
        plt.ylabel('Feature: ' + self.features_name[y_axis])
        plt.legend()
        plt.show()

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = 'nb_model_' + str(int(round(self.score*10000,1)))
        joblib.dump(self.model, file_name)

    def load_model(self, file_name):
        self.model = joblib.load(file_name)

if __name__ == '__main__':
    files = ["RD_1XST.csv", "RD_2P_P.csv", "RD_2X.csv", "RD_3P.csv", "RD_4P.csv",
             "RD_5P.csv", "RD_6P_P.csv", "RD_7P.csv", "RD_8P.csv", "RD-1P.csv",
             "RDT_1P.csv", "RDT_1RX.csv", "RDT_2P.csv"];

    classifier = naive_bayes_classifier(data_file='RD-RDT DATA ALL.csv',
                                        feature_col_range=[2, 9],
                                        test_size=0.2,
                                        features_degree=1,
                                        features_scaling=True)
    classifier.fit(strategy='ovr')
    # classifier.fit(strategy='ovo')
    # classifier.fit()
    # classifier.save_model()
    # classifier.load_model()

    # for file in files:
    #     classifier.evaluate(data_file=file, feature_col_range=[1, 8])
    # classifier.evaluate(data_file="RD-RDT DATA ALL.csv", feature_col_range=[2, 9])
    # print(classifier.predict(data_file='RD-1P.csv')[:100])
