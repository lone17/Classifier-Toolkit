# Decision Tree Classification

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

class decision_tree_classifier(object):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0, test_size=0.2,
                 feature_col_range=[0, 8], label_col=-1, features_degree=1,
                 include_bias=True, features_scaling=False):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            y = data.iloc[:, label_col].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size)
        elif train_set is not None and val_set is not None:
            self.X_train = train_set.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            self.X_test = val_set.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            self.y_train = train_set.iloc[:, label_col].values
            self.y_test = val_set.iloc[:, label_col].values
        else:
            raise RuntimeError('Missing data')

        self.poly = PolynomialFeatures(features_degree, include_bias=include_bias)
        self.X_train = self.poly.fit_transform(self.X_train)

        self.features_scaling = features_scaling
        self.sc = StandardScaler()
        if features_scaling:
            self.X_train = self.sc.fit_transform(self.X_train)

        self.model = None
        self.cm = None
        self.score = 0

    def fit(self, criterion='entropy', min_samples_split=5, min_impurity_decrease=0.00005):

        # Fitting Decision Tree Classification to the Training set
        print('Fitting the train set...')

        self.model = DecisionTreeClassifier(criterion=criterion,
                                            min_samples_split=min_samples_split,
                                            min_impurity_decrease=min_impurity_decrease)

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
        if self.features_scaling:
            X = self.sc.transform(X)

        accuracy = self.model.score(X, y)
        print('Accuracy: ', accuracy * 100)

        return accuracy

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
        if self.features_scaling:
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
        if self.features_scaling:
            X = self.sc.transform(X)

        return self.model.predict(X)

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = 'dt_model_' + str(int(round(self.score*10000,1)))
        joblib.dump(self.model, file_name)

    def load_model(self, file_name):
        self.model = joblib.load(file_name)

if __name__ == '__main__':
    files = ["RD_1XST.csv", "RD_2P_P.csv", "RD_2X.csv", "RD_3P.csv", "RD_4P.csv",
             "RD_5P.csv", "RD_6P_P.csv", "RD_7P.csv", "RD_8P.csv", "RD-1P.csv",
             "RDT_1P.csv", "RDT_1RX.csv", "RDT_2P.csv"];

    classifier = decision_tree_classifier(data_file='RD-RDT DATA ALL.csv',
                                          feature_col_range=[2, 9],
                                          features_degree=2,
                                          include_bias=True)
    classifier.fit()
    classifier.save_model()

    for file in files:
        classifier.evaluate(data_file=file, feature_col_range=[1, 8])
    classifier.evaluate(data_file="RD-RDT DATA ALL.csv", feature_col_range=[2, 9])

    # for file in ["RD-1P.csv"]:
    #     classifier = decision_tree_classifier(data_file=file)
    #     classifier.fit()
    #     print(classifier.confusion_matrix())
    #     classifier.evaluate(data_file='RD_2P_P.csv')
    #     classifier.evaluate(data_file='RD_2X.csv')
    #     classifier.evaluate(data_file='RD_3P.csv')
    #     classifier.evaluate(data_file='RD_4P.csv')
    #     classifier.evaluate(data_file='RD_5P.csv')
    #     classifier.evaluate(data_file='RD_6P_P.csv')
    #     classifier.evaluate(data_file='RD_7P.csv')
    #     classifier.evaluate(data_file='RD_8P.csv')
    #     classifier.evaluate(data_file='RD-1P.csv')
    #     classifier.evaluate(data_file='RDT_1P.csv')
    #     classifier.evaluate(data_file='RDT_1RX.csv')
    #     classifier.evaluate(data_file='RDT_2P.csv')
