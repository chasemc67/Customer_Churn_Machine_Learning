import numpy as np 		# numpy for nan values
import pandas as pd
import random			# random to create crossfold validation

from scipy import stats as st
from sklearn import preprocessing, tree
from sklearn.model_selection import KFold
import argparse
import numpy as np
import pandas as pd
import signal

from sklearn.neural_network import MLPClassifier

filename = "./train_set.csv"

def clean_data(data):
    data = data.drop('cancellation_request', axis=1)
    data = data.drop('id', axis=1)
    return data


def encode_data(data):
    header = data.columns.values.tolist()
    matrix = np.zeros(data.shape, dtype=np.float64)
    encoders = list()
    for i, name in enumerate(list(data.columns.values)):
        if data[name].dtype in [int, float]:
            matrix[:, i] = data.values[:, i]
            encoders.append(None)
        else:
            encoder = preprocessing.LabelEncoder()
            matrix[:, i] = encoder.fit_transform(
                [str(i) for i in data.values[:, i]])
            encoders.append(encoder)
    matrix[np.isnan(matrix)] = - 1
    return matrix.astype(np.float64), header, encoders


def get_data(filename):
    return encode_data(clean_data(pd.read_csv(filename)))


def split_data(data, header):
    target_index = header.index('renewed')
    y = data[:, target_index]
    X = np.delete(data, target_index, axis=1)
    return X, y


def main():
    global siginfo_message
    siginfo_message = 'Loading data ...'
    data, header, encoders = get_data('./train_set.csv')

    # train model
    folds = data.shape[0]
    correct = np.zeros(folds)
    i = 0
    for train, test in KFold(n_splits=folds).split(data):
        i += 1
        siginfo_message = 'Running fold {} of {}'.format(i, folds)
        X, y = split_data(data[train, :], header)
        X_test, y_test = split_data(data[test, :], header)
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(10,10,10), max_iter=10000, random_state=1)
        clf.fit(X, y)
        correct[i - 1] = clf.score(X_test, y_test)

    #mean=0.7069; sem=0.0082
    print("mean={0:0.4f}; sem={1:0.4f}".format(correct.mean(),
                                               st.sem(correct)))

    # output full model
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(10,10,10), max_iter=10000, random_state=1)
    clf.fit(* split_data(data, header))

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