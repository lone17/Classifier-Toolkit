from sklearn.externals import joblib
import numpy as np
import pandas as pd

from Classifier import *
from NeuralNetworkClassifier import *
from util import *

a = np.arange(7*10).reshape(-1, 7)
# df = pd.DataFrame(data=a)
# print(df.values)
model = NeuralNetworkClassifier.load_model('test')
p = model.predict(X = a)
print(p)
