import numpy as np
import pandas as pd
import pickle

import types
import tempfile
import keras.models

from Classifier import *
from NeuralNetworkClassifier import *
# np.random.seed(17)
np.set_printoptions(threshold=np.nan)

def merge_data(files, header=0):
    data = pd.DataFrame()
    for file in files:
        data =  pd.concat([data, pd.read_csv(file, header=header)])

    return data.values

files = ["RD_1XST.csv", "RD_2P_P.csv", "RD_2X.csv", "RD_3P.csv", "RD_4P.csv",
         "RD_5P.csv", "RD_6P_P.csv", "RD_7P.csv", "RD_8P.csv", "RD-1P.csv",
         "RDT_1P.csv", "RDT_1RX.csv", "RDT_2P.csv", "RD-RDT DATA ALL.csv"];

group1 = ['RD_1XST.csv', 'RD_2X.csv', 'RD_3P.csv', 'RD_4P.csv']
group2 = ['RDT_2P.csv', 'RD_2P_P.csv', 'RD_7P.csv', 'RDT_1P.csv']
group3 = ['RD-1P.csv', 'RD_5P.csv', 'RD_6P_P.csv', 'RD_8P.csv']


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

def load(file_name):
    if os.path.isdir(file_name):
        return NeuralNetworkClassifier.load(file_name)

    return Classifier.load(file_name)
