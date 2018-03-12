import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras import backend as K

import pickle

from Classifier import *

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

np.set_printoptions(threshold=np.nan)

num_labels = 2
feature_col_range = [1, 9]
label_col = -1

class BinaryNeuralNetworkClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, features_degree=1, target=2):

        if len(feature_col_range) != 2 or feature_col_range[0] > feature_col_range[1]:
            raise ValueError('Invalid feature_col_range')

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, feature_col_range[0]:feature_col_range[1]].values
            y = data.iloc[:, label_col].values
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size=test_size)
        elif train_set is not None and val_set is not None:
            np.random.shuffle(train_set)
            tmp = train_set[np.where(np.isin(train_set[:,-1], [1, 2, 4, 6, 7, 8, 9, 10]))]
            # tmp = np.append(tmp, train_set[np.where(train_set[:,-1] == 4)][:400], axis=0)
            # tmp = np.append(tmp, train_set[np.where(train_set[:,-1] == 6)][:400], axis=0)
            tmp = np.append(tmp, train_set[np.where(train_set[:,-1] == 0)][:2000], axis=0)
            tmp = np.append(tmp, train_set[np.where(train_set[:,-1] == 3)][:2000], axis=0)
            # tmp = np.append(tmp, train_set[np.where(train_set[:,-1] == 3)][:1000], axis=0)
            # tmp = np.append(tmp, train_set[np.where(train_set[:,-1] == 5)][:15000], axis=0)
            # tmp = np.append(tmp, train_set[np.where(train_set[:,-1] == 7)][:400], axis=0)
            train_set = tmp
            np.random.shuffle(val_set)
            val_set = val_set[:10000]
            for i in range(11):
                print(len(train_set[np.where(train_set[:,-1] == i)]))
            self.X_train = train_set[:, feature_col_range[0]:feature_col_range[1]]
            self.X_test = val_set[:, feature_col_range[0]:feature_col_range[1]]
            self.y_train = train_set[:, label_col]
            self.y_test = val_set[:, label_col]
        else:
            raise RuntimeError('Missing data')

        self.poly = PolynomialFeatures(features_degree, include_bias=True)
        self.X_train = self.poly.fit_transform(self.X_train)

        self.sc = StandardScaler()

        self.model = None
        self.cm = None
        self.score = 0

        self.his = None
        self.num_samples, self.num_features = self.X_train.shape

        self.sc.fit(self.X_train)

        self.target = target
        self.y_train = self.y_train == target

        self.structure()

    def structure(self, hidden_layers=[30], activation='sigmoid'):

        self.model = Sequential()

        self.model.add(Dense(activation=activation,
                             input_dim=self.num_features,
                             units=hidden_layers[0],
                             kernel_initializer=glorot_uniform(),
                             kernel_regularizer=l2(0.0),
                             use_bias=False))

        for layer in hidden_layers[1:]:
            self.model.add(Dense(activation=activation,
                                 units=layer,
                                 kernel_initializer=glorot_uniform(),
                                 kernel_regularizer=l2(0.0),
                                 use_bias=False))

        self.model.add(Dense(activation=activation,
                             units=num_labels,
                             kernel_initializer=glorot_uniform(),
                             kernel_regularizer=l2(0.0),
                             use_bias=False))

    def __transform(self, X, y):
        X = self.poly.transform(X)
        X = self.sc.transform(X)
        y = to_categorical(y == self.target, num_labels)

        return (X, y)

    def acc(self, y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(y_true[:, 0] * (K.reduce_max(y_pred, axis=0)))
        possible_positives = K.sum(y_true[:, 0])
        recall = true_positives / possible_positives
        return recall

    def train_backprop(self, batch_size=1, num_epochs=20, optimizer='adamax',
                       learning_rate=None):

        if learning_rate is not None:
            if optimizer == 'sgd':
                optimizer = keras.optimizers.SGD(lr=learning_rate)
            elif optimizer == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(lr=learning_rate)
            elif optimizer == 'adagrad':
                optimizer = keras.optimizers.Adagrad(lr=learning_rate)
            elif optimizer == 'adadelta':
                optimizer = keras.optimizers.Adadelta(lr=learning_rate)
            elif optimizer == 'adam':
                optimizer = keras.optimizers.Adam(lr=learning_rate)
            elif optimizer == 'adamax':
                optimizer = keras.optimizers.Adamax(lr=learning_rate)
            elif optimizer == 'nadam':
                optimizer = keras.optimizers.Nadam(lr=learning_rate)

        if batch_size is None:
            batch_size = self.num_samples

        print('\nTraining Neural Network with back propagation...\n')

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                           metrics=['accuracy'])

        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=2, verbose=0, mode='auto')

        self.his = self.model.fit(self.sc.transform(self.X_train),
                                  to_categorical(self.y_train, num_labels),
                                  batch_size=batch_size,
                                  epochs=num_epochs,
                                  validation_data=self.__transform(self.X_test, self.y_test),
                                  verbose=1).history

        train_accuracy = self.his['acc'][-1]
        train_loss = self.his['loss'][-1]
        print('\nTrain Set Accuracy: ', train_accuracy * 100, '  Loss: ', train_loss)

        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            X = self.poly.transform(self.X_test)
            X = self.sc.transform(X)
            y_pred = self.model.predict(X).argmax(axis=1)
            [_, self.score] = self.evaluate(X=self.X_test, y=self.y_test)

    def train_evolstrategy(self, batch_size=None, num_epochs=1000, population=100,
                           sigma=0.01, learning_rate=0.001, boosting_ops=0,
                           optimizer='adamax'):

        if batch_size is None:
            batch_size = self.num_samples

        print('\nTraining Neural Network with evolution strategy...\n')

        self.model.compile(optimizer='adamax',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.his = {'acc':[], 'loss':[], 'val_acc':[], 'val_loss':[]}

        num_w = 0
        w = []
        for layer in self.model.layers:
            cur_w = layer.get_weights()[0]
            shape = cur_w.shape
            num_w += shape[0] * shape[1]
            w = np.concatenate((w, np.ravel(cur_w)))

        X = self.sc.transform(self.X_train)
        y = to_categorical(self.y_train, num_labels)
        X_val, y_val = self.__transform(self.X_test, self.y_test)

        r = 17
        for i in range(num_epochs):
            N = np.random.randn(population, num_w)
            R = np.zeros(population)
            for j in range(population):
                w_try = w + sigma * N[j]
                # [loss, acc] = self.__result(w_try, X, y)
                loss = -self.__precision(w_try, X, y)
                R[j] = -loss
            A = (R - np.mean(R)) / np.std(R)
            w = w + learning_rate / (population*sigma) * np.dot(N.T, A)

            [loss, acc] = self.__result(w, X, y)
            [val_loss, val_acc] = self.__result(w, X_val, y_val)

            self.his['loss'].append(loss)
            self.his['acc'].append(acc)
            self.his['val_loss'].append(val_loss)
            self.his['val_acc'].append(val_acc)
            print('epoch %d/%d. loss: %f, accuracy: %f' % (i+1, num_epochs, loss, acc*100))
            r = acc

        if (boosting_ops > 0):
            print('\nBoosting accuracy...\n')

            for i in range(boosting_ops):
                N = np.random.randn(population, num_w)
                new_w = w.copy()
                his_tmp = None
                for j in range(population):
                    w_try = w + sigma * N[j]
                    [loss, acc] = self.__result(w_try, X, y)
                    if acc > r:
                        r = acc
                        new_w = w_try
                w = new_w
                [loss, acc] = self.__result(w, X, y)
                print('round %d/%d. loss: %f, accuracy: %f' % (i+1, 100, loss, acc*100))

        print('\nTrain Set Accuracy: ', acc * 100, '  Loss: ', loss)

        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            y_pred = self.model.predict(X_val).argmax(axis=1)
            [_, self.score] = self.evaluate(X=self.X_test, y=self.y_test)

    def __fold(self, w):
        weights = []
        for layer in self.model.layers:
            shape = layer.get_weights()[0].shape
            elements = shape[0] * shape[1]
            weights.append(np.reshape(w[:elements], shape))
            w = w[elements:]
        return weights

    def __result(self, w, X, y):
        weights = self.__fold(w)
        for w, layer in zip(weights, self.model.layers):
            layer.set_weights([w])

        res = self.model.evaluate(X, y, batch_size=self.num_samples, verbose=0)

        return res

    def __precision(self, w, X, y):
        weights = self.__fold(w)
        for w, layer in zip(weights, self.model.layers):
            layer.set_weights([w])

        pred = (self.model.predict(X).argmax(axis=1))

        return f1_score(y.argmax(axis=1), pred)

    def confusion_matrix(self):
        return self.cm

    def evaluate(self, X=None, y=None, data_file=None, header=0,
                 batch_size=None):

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

        if batch_size is None:
            batch_size = X.shape[0]
            print(batch_size)

        X = self.poly.transform(X)
        X = self.sc.transform(X)
        y = y == self.target
        y_matrix = to_categorical(y, num_labels)

        [loss, accuracy] = self.model.evaluate(X, y_matrix, batch_size=batch_size)
        print('Accuracy: ', accuracy * 100, '  Loss: ', loss)

        pred = (self.model.predict(X).argmax(axis=1))
        # new_acc = np.count_nonzero(y == pred) / len(y) * 100
        # print('Accuracy after post-processing: ', new_acc)

        self.cm = confusion_matrix(y, pred)
        print('recall: ', recall_score(y, pred))
        print('precision: ', precision_score(y, pred))
        print('f1 score: ', f1_score(y, pred))

        return [loss, accuracy]

    def probality(self, X=None, data_file=None, header=0):

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
        X = self.sc.transform(X)

        return self.model.predict(X, verbose=1)

    def predict(self, X=None, data_file=None, header=0):

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
        X = self.sc.transform(X)
        prob = self.model.predict(X, verbose=1)

        return self.__post_process(prob.argmax(axis=1))

    def plot(self, name='loss and accuracy per epoch'):
        print(name)
        plt.plot(self.his['loss'], label='train loss')
        plt.plot(self.his['acc'], label='train accuracy')
        plt.plot(self.his['val_loss'], label='evaluate loss')
        plt.plot(self.his['val_acc'], label='evaluate accuracy')
        plt.legend()
        plt.title(name)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def save_weights(self, file_name=None):
        if file_name is None:
            file_name = 'nn_weights_' + str(int(round(self.score*10000,1)))
        self.model.save_weights(file_name)

    def load_weights(self, file_name):
        self.model.load_weights(file_name)
        self.model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
        # cnt = 0
        # for layer in self.model.layers:
        #     print(layer.get_weights()[0].shape)
        #     for w in np.ravel(layer.get_weights()):
        #         cnt = cnt+1
        #         if abs(w) < 0.01:
        #             print(w)
        # print(cnt)

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = 'nn_model_' + str(int(round(self.score*10000,1)))
        self.model.save(file_name)

if __name__ == '__main__':

    classifier = BinaryNeuralNetworkClassifier(
                                         # data_file='RD-RDT DATA ALL.csv',
                                         train_set=merge_data(group3 + group2),
                                         val_set=merge_data(group1),
                                         )
    classifier.structure()
    classifier.train_backprop()
    # classifier.train_evolstrategy()
    print(classifier.cm)

    # classifier = BinaryNeuralNetworkClassifier(
    #                                      # data_file='RD-RDT DATA ALL.csv',
    #                                      train_set=merge_data(group3 + group1),
    #                                      val_set=merge_data(group2),
    #                                      )
    # classifier.structure()
    # classifier.train_backprop()
    # # classifier.train_evolstrategy(num_epochs=1000)
    # print(classifier.cm)
    #
    # classifier = BinaryNeuralNetworkClassifier(
    #                                      # data_file='RD-1P.csv',
    #                                      train_set=merge_data(group1 + group2),
    #                                      val_set=merge_data(group3),
    #                                      )
    # classifier.structure()
    # classifier.train_backprop()
    # # classifier.train_evolstrategy(num_epochs=1000)
    # print(classifier.cm)
    # # classifier.save_weights()


    for file in files:
        classifier.evaluate(data_file=file)
    # classifier.plot(name='[2, 9]')
