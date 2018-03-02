# Decision Tree Classification

# Importing the libraries
from sklearn.tree import DecisionTreeClassifier as Model

from Classifier import *

class DecisionTreeClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, feature_col_range=[1, 9], label_col=-1,
                 features_degree=1):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                            test_size, feature_col_range, label_col,
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
        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            y_pred = self.predict(self.X_test)
            self.score = self.evaluate(X=self.X_test, y=self.y_test)

            # Making the Confusion Matrix
            self.cm = confusion_matrix(self.y_test, y_pred, labels=[i for i in range(11)])

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = 'dt_model_' + str(int(round(self.score*10000,1)))
        joblib.dump(self.model, file_name)

    def load_model(self, file_name):
        self.model = joblib.load(file_name)

if __name__ == '__main__':

    classifier = DecisionTreeClassifier(
                                        # data_file='RD-RDT DATA ALL.csv',
                                        train_set=merge_data(group3 + group2),
                                        val_set=merge_data(group1),
                                        )
    classifier.fit()
    classifier = DecisionTreeClassifier(
                                        # data_file='RD-RDT DATA ALL.csv',
                                        train_set=merge_data(group1 + group3),
                                        val_set=merge_data(group2),
                                        )
    classifier.fit()
    classifier = DecisionTreeClassifier(
                                        # data_file='RD-RDT DATA ALL.csv',
                                        train_set=merge_data(group1 + group2),
                                        val_set=merge_data(group3),
                                        )
    classifier.fit()
    # classifier.evaluate(data_file='RD_1XST.csv')
    # print(classifier.cm)
    # classifier.save_model()
    # classifier.load_model('dt_model_8394')

    # for file in files:
    #     classifier.evaluate(data_file=file)
