# K-Nearest Neighbors (K-NN)

# Importing the libraries
from sklearn.neighbors import KNeighborsClassifier as Model

from Classifier import *

class KNearestNeighborsClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, features_col_range=[1, 8], label_col=-1,
                 features_degree=1):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                            test_size, features_col_range, label_col,
                            features_degree, True)

    def fit(self, num_neighbors=100, p=1):

        print('Using K-Nearest Neighbors Classifier...')

        self.model = Model(n_neighbors=num_neighbors, metric='minkowski', p=p)

        self.model.fit(self.X_train, self.y_train)

        print('\nTrain Set Accuracy: ', self.model.score(self.X_train, self.y_train) * 100)

        self.evaluate_test()

    def save(self, file_name=None):
        del self.X_train
        del self.y_train
        del self.X_test
        del self.y_test
        
        if file_name is None:
            file_name = 'knn_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)
