# Decision Tree Classification

# Importing the libraries
from sklearn.tree import DecisionTreeClassifier as Model

from Classifier import *

class DecisionTreeClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, features_col_range=[0, 8], label_col=-1,
                 features_degree=1):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                            test_size, features_col_range, label_col,
                            features_degree, True)

    def fit(self, criterion='entropy', min_samples_split=5, min_impurity_decrease=0.01):

        # Fitting Decision Tree Classification to the Training set
        print('Using Decision Tree Classifier...')

        self.model = Model(criterion=criterion,
                           min_samples_split=min_samples_split,
                           min_impurity_decrease=min_impurity_decrease)

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
            file_name = 'dt_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)
