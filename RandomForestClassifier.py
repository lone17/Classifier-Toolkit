# Random Forest Classification

# Importing the libraries
from sklearn.ensemble import RandomForestClassifier as Model

from Classifier import *

class RandomForestClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, features_col_range=[0, 8], label_col=-1,
                 features_degree=1):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                            test_size, features_col_range, label_col,
                            features_degree, True)

    def fit(self, num_trees=80, criterion='entropy', min_samples_split=5,
            min_impurity_decrease=0.0003):

        print('Using Random Forest Classifier...')

        self.model = Model(n_estimators=num_trees,
                           criterion=criterion,
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
            file_name = 'rf_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)
