from util import *

from NeuralNetworkClassifier import *
Model = NeuralNetworkClassifier

sets = [(group3 + group2, group1),
        (group3 + group1, group2),
        (group1 + group2, group3)]

# for s in sets:
#     classifier = Model(
#         train_set=merge_data(s[0]),
#         val_set=merge_data(s[1]),
#         features_col_range=[0, 8],
#         features_degree=3
#     )
#
#     # classifier.structure()
#     classifier.train_backprop()
#     # classifier.train_evolution(num_epochs=1000)
#
#     print(classifier.cm)
#     for file in ['RD-RDT DATA ALL.csv']:
#           classifier.evaluate(data_file=file)
#           print(classifier.cm)
    # classifier.save()
    # classifier.plot()

classifier = load('nn_model_7077')
for file in files:
    classifier.evaluate(data_file=file)
    print(classifier.cm)
