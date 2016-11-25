
from scipy import stats as st
from sklearn import preprocessing, tree
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import signal

MAX_DECISION_TREE_DEPTH = 5

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
    global siginfo_message
    siginfo_message = 'Loading data ...'
    data, header, encoders = get_data('Data/train_set.csv')

    # train model
    loo = LeaveOneOut()
    correct = 0
    i = 0
    for train, test in loo.split(data):
        i += 1
        siginfo_message = 'Running split {} of {}'.format(i, data.shape[0])
        X, y = split_data(data[train, :], header)
        X_test, y_test = split_data(data[test, :], header)
        clf = tree.DecisionTreeClassifier(criterion='entropy',
                                          max_depth=MAX_DECISION_TREE_DEPTH)
        clf.fit(X, y)
        correct += clf.score(X_test, y_test)
    print("Accuracy: {0:0.4f}".format(correct / data.shape[0]))

    # output full model
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=MAX_DECISION_TREE_DEPTH)
    clf.fit(* split_data(data, header))
    with open('tree.dot', 'w') as outfile:
        feature_names = header[:header.index('renewed')] + \
            header[header.index('renewed') + 1:]
        tree.export_graphviz(clf,
                             feature_names=feature_names,
                             filled=True,
                             out_file=outfile)

if __name__ == '__main__':
    # setup numpy
    np.seterr(all='raise')  # don't ignore errors

    # setup siginfo response system
    global siginfo_message
    siginfo_message = None
    if hasattr(signal, 'SIGINFO'):
        signal.signal(signal.SIGINFO,
                      lambda signum, frame: print(siginfo_message))

    # run
    main()
