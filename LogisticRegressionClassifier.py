# Kernel SVM

# Importing the libraries
from sklearn.linear_model import LogisticRegression

from Classifier import *

class LogisticRegressionClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, features_col_range=[1, 8], label_col=-1,
                 features_degree=3):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                            test_size, features_col_range, label_col,
                            features_degree, True)

    def fit(self, C=20, max_iter=10000, solver='liblinear'):

        print('Using Logistic Regression Classfier...')

        estimator = LogisticRegression(C=C, solver=solver, max_iter=max_iter,
                                       verbose=0)

        self.model = OneVsRestClassifier(estimator=estimator)

        self.model.fit(self.X_train, self.y_train)

        print('\nTrain Set Accuracy: ', self.model.score(self.X_train, self.y_train) * 100)

        # Predicting the Test set results
        self.evaluate_test()

    def save(self, file_name=None):
        del self.X_train
        del self.y_train
        del self.X_test
        del self.y_test
        
        if file_name is None:
            file_name = 'lr_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)
