# Kernel SVM

# Importing the libraries
from sklearn.linear_model import LogisticRegression

from Classifier import *

class LogisticRegressionClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, feature_col_range=[2, 9], label_col=-1,
                 features_degree=3):

        Classifier.__init__(self, train_set, val_set, data_file, header,
                           test_size, feature_col_range, label_col,
                           features_degree, True)

    def fit(self, C=20, max_iter=10000, strategy='ovo', solver='liblinear'):

        print('Using Logistic Regression Classfier...')

        estimator = LogisticRegression(C=C, solver=solver, max_iter=max_iter,
                                       verbose=0)

        if strategy == 'ovo':
            self.model = OneVsOneClassifier(estimator=estimator)
        elif strategy == 'ovr':
            self.model = OneVsRestClassifier(estimator=estimator)
        else:
            raise RuntimeError('strategy can only be \'ovo\' or \'ovr\'')

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

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = 'lr_model_' + str(int(round(self.score*10000,1)))
        joblib.dump(self.model, file_name)

if __name__ == '__main__':

    classifier = LogisticRegressionClassifier(
                                              # data_file='RD-RDT DATA ALL.csv',
                                              train_set=merge_data(group3 + group2),
                                              val_set=merge_data(group1),
                                              )
    classifier.fit()
    classifier = LogisticRegressionClassifier(
                                              # data_file='RD-RDT DATA ALL.csv',
                                              train_set=merge_data(group1 + group3),
                                              val_set=merge_data(group2),
                                              )
    classifier.fit()
    classifier = LogisticRegressionClassifier(
                                              # data_file='RD-RDT DATA ALL.csv',
                                              train_set=merge_data(group1 + group2),
                                              val_set=merge_data(group3),
                                              )
    classifier.fit()
    # classifier.plot(2,5)
    # classifier.save_model()
    # classifier.load_model('lr_model_7382')

    # for file in files:
    #     classifier.evaluate(data_file=file)
