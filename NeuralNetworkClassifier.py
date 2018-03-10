import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras.regularizers import l2

from Classifier import *

from statistics import mean

num_labels = 11

class NeuralNetworkClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=0,
                 test_size=0.2, features_col_range=[3, 8], label_col=-1,
                 features_degree=2):

        if len(features_col_range) != 2 or features_col_range[0] > features_col_range[1]:
            raise ValueError('Invalid features_col_range')
        self.features_col_range = features_col_range
        self.label_col = label_col

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, features_col_range[0]:features_col_range[1]].values
            y = data.iloc[:, label_col].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        elif train_set is not None and val_set is not None:
            # train_set = train_set[np.where(train_set[:,-1] != 5)]
            # val_set = val_set[np.where(val_set[:,-1] != 5)]
            self.X_train = train_set[:, features_col_range[0]:features_col_range[1]]
            self.X_test = val_set[:, features_col_range[0]:features_col_range[1]]
            self.y_train = train_set[:, label_col].astype('int')
            self.y_test = val_set[:, label_col].astype('int')
        else:
            raise RuntimeError('Missing data')

        self.poly = PolynomialFeatures(features_degree, include_bias=True)
        self.X_train = self.poly.fit_transform(self.X_train)
        self.X_test = self.poly.transform(self.X_test)

        self.model = None
        self.cm = None
        self.score = 0

        self.his = None
        self.num_samples, self.num_features = self.X_train.shape

        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)

        self.structure()

    def structure(self, hidden_layers=[30, 30], activation='sigmoid'):

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

    def train_backprop(self, batch_size=None, num_epochs=10000, optimizer='adamax',
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

        X, y = self.X_train, to_categorical(self.y_train, num_labels)
        X_val, y_val = self.X_test, to_categorical(self.y_test, num_labels)

        # self.model.fit(X, y,
        #                batch_size=1,
        #                epochs=20,
        #                validation_data=(X_val, y_val),
        #                verbose=1)
        self.his = self.model.fit(X, y,
                                  batch_size=batch_size,
                                  epochs=num_epochs,
                                  validation_data=(X_val, y_val),
                                  callbacks=[es],
                                  verbose=1).history

        train_accuracy = self.his['acc'][-1]
        train_loss = self.his['loss'][-1]
        print('\nTrain Set Accuracy: ', train_accuracy * 100, '  Loss: ', train_loss)

        self.__evaluate_test(batch_size)


    def train_evolution(self, batch_size=None, num_epochs=1000, population = 100,
                           sigma = 0.01, learning_rate = 0.001, boosting_ops = 0,
                           optimizer='adamax'):

        if batch_size is None:
            batch_size = self.num_samples

        print('\nTraining Neural Network with evolution strategy...\n')

        self.model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])

        self.his = {'acc':[], 'loss':[], 'val_acc':[], 'val_loss':[]}

        num_w = 0
        w = []
        for layer in self.model.layers:
            cur_w = layer.get_weights()[0]
            shape = cur_w.shape
            num_w += shape[0] * shape[1]
            w = np.concatenate((w, np.ravel(cur_w)))

        X, y = self.X_train, to_categorical(self.y_train, num_labels)
        X_val, y_val = self.X_test, to_categorical(self.y_test, num_labels)

        r = 17
        for i in range(num_epochs):
            N = np.random.randn(population, num_w)
            R = np.zeros(population)
            for j in range(population):
                w_try = w + sigma * N[j]
                self.__set_weights(w_try)
                [loss, acc] = self.__result(X, y)
                R[j] = -loss
            A = (R - np.mean(R)) / np.std(R)
            old_w = w
            w += learning_rate / (population*sigma) * np.dot(N.T, A)

            self.__set_weights(w)
            [loss, acc] = self.__result(X, y)
            [val_loss, val_acc] = self.__result(X_val, y_val)

            self.his['loss'].append(loss)
            self.his['acc'].append(acc)
            self.his['val_loss'].append(val_loss)
            self.his['val_acc'].append(val_acc)
            print('epoch %d/%d. loss: %f, accuracy: %f' % (i+1, num_epochs, loss, acc*100))
            r = acc

            if val_loss > mean(self.his['val_loss'][-200:]):
                w = old_w
                self.__set_weights(w)
                break

        if boosting_ops > 0:
            print('\nBoosting accuracy...\n')

            for i in range(boosting_ops):
                N = np.random.randn(population, num_w)
                new_w = w.copy()
                his_tmp = None
                for j in range(population):
                    w_try = w + sigma * N[j]
                    self.__set_weights(w)
                    [loss, acc] = self.__result(X, y)
                    if acc > r:
                        r = acc
                        new_w = w_try
                w = new_w
                self.__set_weights(w)
                [loss, acc] = self.__result(X, y)
                print('round %d/%d. loss: %f, accuracy: %f' % (i+1, 100, loss, acc*100))

        print('\nTrain Set Accuracy: ', acc * 100, '  Loss: ', loss)

        self.__evaluate_test(batch_size)

    def __fold(self, w):
        weights = []
        for layer in self.model.layers:
            shape = layer.get_weights()[0].shape
            elements = shape[0] * shape[1]
            weights.append(np.reshape(w[:elements], shape))
            w = w[elements:]
        return weights

    def __set_weights(self, w):
        weights = self.__fold(w)
        for w, layer in zip(weights, self.model.layers):
            layer.set_weights([w])

    def __result(self, X, y):
        return self.model.evaluate(X, y, batch_size=self.num_samples, verbose=0)

    def __evaluate_test(self, batch_size):
        if len(self.X_test) > 0:
            print('\nEvaluating on test set...')
            X = self.X_test
            y = self.y_test
            y_matrix = to_categorical(self.y_test, num_labels)

            [loss, accuracy] = self.model.evaluate(X, y_matrix, batch_size=batch_size)
            print('Accuracy: ', accuracy * 100, '  Loss: ', loss)

            pred = self.post_process(self.model.predict(X).argmax(axis=1))
            accuracy = np.count_nonzero(y == pred) / len(y)
            print('Accuracy after post-processing: ', accuracy * 100)

            self.cm = confusion_matrix(y, pred, labels=[i for i in range(num_labels)])

            self.score = accuracy

    def confusion_matrix(self):
        return self.cm

    def evaluate(self, X=None, y=None, data_file=None, header=0,
                 batch_size=None):

        features_col_range = self.features_col_range
        label_col = self.label_col

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            # data = data[np.where(data[:,-1] != 5)]
            print('\nEvaluating on ', data_file, '...', sep='')
            X = data[:, features_col_range[0]:features_col_range[1]]
            y = data[:, label_col]
        elif X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
        else:
            raise RuntimeError('Missing data')

        if batch_size is None:
            batch_size = X.shape[0]

        X = self.poly.transform(X)
        X = self.sc.transform(X)
        y_matrix = to_categorical(y, num_labels)

        [loss, accuracy] = self.model.evaluate(X, y_matrix, batch_size=batch_size)
        print('Accuracy: ', accuracy * 100, '  Loss: ', loss)

        pred = self.post_process(self.model.predict(X).argmax(axis=1))
        accuracy = np.count_nonzero(y == pred) / len(y)
        print('Accuracy after post-processing: ', accuracy * 100)

        self.cm = confusion_matrix(y, pred, labels=[i for i in range(num_labels)])

        return [loss, accuracy]

    def probality(self, X=None, data_file=None, header=0):

        features_col_range = self.features_col_range

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, features_col_range[0]:features_col_range[1]].values
        elif X is not None:
            X = np.array(X[:, features_col_range[0]:features_col_range[1]])
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        X = self.sc.transform(X)

        return self.model.predict(X, verbose=1)

    def predict(self, X=None, data_file=None, header=0):

        features_col_range = self.features_col_range

        if data_file is not None:
            data = pd.read_csv(data_file, header=header)
            X = data.iloc[:, features_col_range[0]:features_col_range[1]].values
        elif X is not None:
            X = np.array(X[:, features_col_range[0]:features_col_range[1]])
        else:
            raise RuntimeError('Missing data')

        X = self.poly.transform(X)
        X = self.sc.transform(X)
        prob = self.model.predict(X, verbose=1)

        return self.post_process(prob.argmax(axis=1))

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

    def save(self, model_dir=None):
        del self.X_train
        del self.y_train
        del self.X_test
        del self.y_test

        if model_dir is None:
            model_dir = 'nn_model_' + str(int(round(self.score*10000,1)))

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # save weights
        self.model.save_weights(os.path.join(model_dir, 'weights'))

        # save save architecture
        self.model.save(os.path.join(model_dir, 'architecture'))

        self.model = None
        joblib.dump(self, os.path.join(model_dir, 'model'))

    @classmethod
    def load(NeuralNetworkClassifier, model_dir):
        classifier = joblib.load(os.path.join(model_dir, 'model'))
        classifier.model = load_model(os.path.join(model_dir, 'architecture'))
        classifier.model.load_weights(os.path.join(model_dir, 'weights'))
        classifier.model.compile(optimizer='adamax', loss='categorical_crossentropy',
                                 metrics=['accuracy'])

        return classifier

if __name__ == '__main__':

    # classifier = NeuralNetworkClassifier(
    #                                      # data_file='RD-RDT DATA ALL.csv',
    #                                      train_set=merge_data(group3 + group2),
    #                                      val_set=merge_data(group1),
    #                                      )
    # classifier.structure()
    # # classifier.train_backprop()
    # classifier.train_evolution(num_epochs=1000)
    # print(classifier.cm)
    # for file in files:
    #     classifier.evaluate(data_file=file)
    #     print(classifier.cm)
    # classifier.save()




    classifier = NeuralNetworkClassifier(
                                         # data_file='RD-RDT DATA ALL.csv',
                                         train_set=merge_data(group3 + group1),
                                         val_set=merge_data(group2),
                                         )
    classifier.structure()
    classifier.train_backprop()
    # classifier.train_evolution(num_epochs=1000)
    print(classifier.cm)
    for file in files:
        classifier.evaluate(data_file=file)
        print(classifier.cm)
    classifier.plot()
    # classifier.save()




    # classifier = NeuralNetworkClassifier(
    #                                      # data_file='RD-1P.csv',
    #                                      train_set=merge_data(group1 + group2),
    #                                      val_set=merge_data(group3),
    #                                      )
    # classifier.structure()
    # # classifier.train_backprop()
    # classifier.train_evolution(num_epochs=1000)
    # print(classifier.cm)
    # for file in files:
    #     classifier.evaluate(data_file=file)
    #     print(classifier.cm)
    # classifier.save()

    # classifier = NeuralNetworkClassifier.load('nn_model_7379')
    # for file in files:
    #     classifier.evaluate(data_file=file)
    #     print(classifier.cm)
