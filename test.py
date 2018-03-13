from util import *

from DecisionTreeClassifier import *
from RandomForestClassifier import *
from KNearestNeighborsClassifier import *
from LogisticRegressionClassifier import *
from NeuralNetworkClassifier import *

# classifier = load('rf_model_7580')
# classifier = load('lr_model_7230')
# classifier = load('knn_model_7177')
# classifier = load('dt_model_7173')
classifier = load('nn_model_7077')

X = np.arange(8*10).reshape(-1, 8)

p = classifier.predict(X=X)
print(p)

prob = classifier.probality(X=X)
print(prob)

e = classifier.evaluate(X=X, y=p)
print(e)
print(classifier.confusion_matrix())
