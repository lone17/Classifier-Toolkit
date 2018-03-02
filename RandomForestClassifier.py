# Random Forest Classification

# Importing the libraries
from sklearn.ensemble import RandomForestClassifier as Model

from Classifier import *

class RandomForestClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, feature_col_range=[1, 9], label_col=-1,
                 features_degree=1):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                            test_size, feature_col_range, label_col,
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
        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            y_pred = self.predict(self.X_test)
            self.score = self.evaluate(X=self.X_test, y=self.y_test)

            # Making the Confusion Matrix
            self.cm = confusion_matrix(self.y_test, y_pred, labels=[i for i in range(11)])

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = 'rf_model_' + str(int(round(self.score*10000,1)))
        joblib.dump(self.model, file_name)


if __name__ == '__main__':

    classifier = RandomForestClassifier(
                                        # data_file='RD-RDT DATA ALL.csv',
                                        train_set=merge_data(group3 + group2),
                                        val_set=merge_data(group1),
                                        )
    classifier.fit()
    classifier = RandomForestClassifier(
                                        # data_file='RD-RDT DATA ALL.csv',
                                        train_set=merge_data(group1 + group3),
                                        val_set=merge_data(group2),
                                        )
    classifier.fit()
    classifier = RandomForestClassifier(
                                        # data_file='RD-RDT DATA ALL.csv',
                                        train_set=merge_data(group1 + group2),
                                        val_set=merge_data(group3),
                                        )
    classifier.fit()
    # classifier.save_model()
    # classifier.load_model('rf_model_8629')

    # for file in files:
    #     classifier.evaluate(data_file=file)