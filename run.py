from util import *

from DecisionTreeClassifier import *
from RandomForestClassifier import *
from KNearestNeighborsClassifier import *
from LogisticRegressionClassifier import *
Model = DecisionTreeClassifier
# Model = RandomForestClassifier
# Model = KNearestNeighborsClassifier
# Model = LogisticRegressionClassifier

sets = [(group3 + group2, group1),
        (group3 + group1, group2),
        (group1 + group2, group3)]

# for s in sets[1:2]:
#     classifier = Model(
#         train_set=merge_data(s[0]),
#         val_set=merge_data(s[1]),
#         # features_col_range=[2, 8],
#         # features_degree=3
#     )
#
#     classifier.fit()
#
#     print(classifier.cm)
#     for file in ['RD-RDT DATA ALL.csv']:
#           classifier.evaluate(data_file=file)
#           print(classifier.cm)
#     classifier.save()

# classifier = load('rf_model_7580')
# for file in files:
#     classifier.evaluate(data_file=file)
#     print(classifier.cm)
