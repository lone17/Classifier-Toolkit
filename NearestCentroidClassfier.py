# Nearest Centroid Classifier

# Importing the libraries
from sklearn.neighbors import NearestCentroid as Model

from Classifier import *

class NearestCentroidClassfier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, feature_col_range=[2, 9], label_col=-1,
                 features_degree=3):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                            test_size, feature_col_range, label_col,
                            features_degree, True)

    def fit(self, metric='manhattan'):

        print('Using Nearest Centroid Classfier...')

        self.model = Model(metric=metric, shrink_threshold=-17)

        self.model.fit(self.X_train, self.y_train)

        print('\nTrain Set Accuracy: ', self.model.score(self.X_train, self.y_train) * 100)

        # Predicting the Test set results
        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            y_pred = self.predict(self.X_test)
            self.score = self.evaluate(X=self.X_test, y=self.y_test)

            # Making the Confusion Matrix
            self.cm = confusion_matrix(self.y_test, y_pred,
                                       labels=[i for i in range(num_labels)])

    def probality(self, X=None, data_file=None, header=0, feature_col_range=[1, 9]):
        print('probality() is not supported')
        return None

if __name__ == '__main__':

    classifier = NearestCentroidClassfier(
                                         # data_file='RD-RDT DATA ALL.csv',
                                         train_set=merge_data(group3 + group2),
                                         val_set=merge_data(group1),
                                         )
    classifier.fit()
    classifier = NearestCentroidClassfier(
                                         # data_file='RD-RDT DATA ALL.csv',
                                         train_set=merge_data(group1 + group3),
                                         val_set=merge_data(group2),
                                         )
    classifier.fit()
    classifier = NearestCentroidClassfier(
                                         # data_file='RD-RDT DATA ALL.csv',
                                         train_set=merge_data(group1 + group2),
                                         val_set=merge_data(group3),
                                         )
    classifier.fit()
    # classifier.plot(2,4)
    # classifier.save_model()
    # classifier.load_model()

    # for file in files:
    #     classifier.evaluate(data_file=file)
