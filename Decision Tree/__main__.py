
from scipy import stats as st
from sklearn import preprocessing, tree
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

MAX_DECISION_TREE_DEPTH = 5
NUMBER_OF_FOLDS = 10

def get_data(filename):
    return encode_data(clean_data(pd.read_csv(filename)))

def encode_data(data):
    header = data.columns.values.tolist()
    matrix = data.as_matrix()
    encoders = list()
    for i in range(matrix.shape[1]):
        encoder = preprocessing.LabelEncoder()
        matrix[:, i] = encoder.fit_transform([str(x) for x in matrix[:, i]])
        encoders.append(encoder)
    return matrix.astype(np.uint64), header, encoders

def clean_data(data):
    data = data.drop('cancellation_request', axis=1)
    return data

def split_data(data, header):
    target_index = header.index('renewed')
    y = data[:, target_index]
    X = np.delete(data, target_index, axis=1)
    return X, y

def main():
    data, header, encoders = get_data('Data/train_set.csv')

    # train model
    X, y = split_data(data, header)
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=MAX_DECISION_TREE_DEPTH)
    correct = cross_val_score(clf, X, y, cv=NUMBER_OF_FOLDS)
    print("Accuracy: mean={0:0.4f}, sem={1:0.4f}".format(
        correct.mean(), st.sem(correct)))

    # output full model
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=MAX_DECISION_TREE_DEPTH)
    clf.fit(X, y)
    with open('tree.dot', 'w') as outfile:
        feature_names = header[:header.index('renewed')] + \
            header[header.index('renewed') + 1:]
        tree.export_graphviz(clf,
                             feature_names=feature_names,
                             filled=True,
                             out_file=outfile)

if __name__ == '__main__':
    main()
