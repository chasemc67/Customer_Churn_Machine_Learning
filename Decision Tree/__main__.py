#!/usr/bin/env python

from scipy import stats as st
from sklearn import preprocessing, tree
from sklearn.model_selection import KFold
import argparse
import numpy as np
import pandas as pd
import signal

TARGET_NAME = 'renewed'


def clean_data(data):
    data = data.drop('id', axis=1)
    return data


def encode_data(data):
    header = data.columns.values.tolist()
    matrix = np.zeros(data.shape, dtype=np.float64)
    for i, name in enumerate(list(data.columns.values)):
        if data[name].dtype in [int, float, bool]:
            matrix[:, i] = data.values[:, i]
        else:
            encoder = preprocessing.LabelEncoder()
            matrix[:, i] = encoder.fit_transform(
                [str(i) for i in data.values[:, i]])
    matrix[np.isnan(matrix)] = - 1
    np.random.shuffle(matrix)
    return matrix, header


def extract_target(data, header):
    target_index = header.index(TARGET_NAME)
    y = data[:, target_index]
    X = np.delete(data, target_index, axis=1)
    return X, y


def get_data(filename):
    train_only, data = split_data(clean_data(pd.read_csv(filename)))
    train_only_matrix, _ = encode_data(train_only)
    matrix, header = encode_data(data)
    return train_only_matrix, matrix, header


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth',
                        type=int,
                        help='maximum depth of tree',
                        default=5)
    parser.add_argument('-f', '--folds',
                        type=int,
                        help='number of folds for data')
    return vars(parser.parse_args())


def split_data(data):
    true_data = data[data['cancellation_request'] == True]
    true_data.drop('cancellation_request', axis=1)
    false_data = data[data['cancellation_request'] == False]
    false_data.drop('cancellation_request', axis=1)
    return true_data, false_data


def main():
    global siginfo_message
    args = parse_args()
    siginfo_message = 'Loading data ...'
    train_only_matrix, matrix, header = get_data('Data/train_set.csv')

    # split train only data
    X_train_only, y_train_only = extract_target(train_only_matrix, header)

    # get cross validation error
    folds = matrix.shape[0] if args['folds'] is None else args['folds']
    correct = np.zeros(folds)
    for i, (train, test) in enumerate(KFold(n_splits=folds).split(matrix)):
        siginfo_message = 'Running fold {} of {}'.format(i + 1, folds)

        # make learner and train on train only data
        clf = tree.DecisionTreeClassifier(criterion='entropy',
                                          max_depth=args['depth'])
        clf.fit(X_train_only, y_train_only)

        # train on cross validation train data
        X, y = extract_target(matrix[train, :], header)
        clf.fit(X, y)

        # get accuracy on test data
        X_test, y_test = extract_target(matrix[test, :], header)
        correct[i - 1] = clf.score(X_test, y_test)
    print("mean={0:0.4f}; sem={1:0.4f}".format(correct.mean(),
                                               st.sem(correct)))

    # output full model
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=args['depth'])
    clf.fit(X_train_only, y_train_only)
    X, y = extract_target(matrix, header)
    clf.fit(X, y)
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
