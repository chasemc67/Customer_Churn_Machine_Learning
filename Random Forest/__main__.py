#!/usr/bin/env python

from scipy import stats as st
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import argparse
import numpy as np
import pandas as pd
import signal

TARGET_NAME = 'renewed'


def clean_data(data):
    data = data[data['cancellation_request'] == False]
    data = data.drop('cancellation_request', axis=1)
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
    return matrix, header


def get_data(filename):
    return encode_data(clean_data(pd.read_csv(filename)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth',
                        type=int,
                        help='maximum depth of tree in random forest',
                        default=7)
    parser.add_argument('-e', '--estimators',
                        type=int,
                        help='number of estimators in random forest',
                        default=12)
    parser.add_argument('-f', '--folds',
                        type=int,
                        help='number of folds for data')
    return vars(parser.parse_args())


def split_data(data, header):
    target_index = header.index(TARGET_NAME)
    y = data[:, target_index]
    X = np.delete(data, target_index, axis=1)
    return X, y


def main():
    global siginfo_message
    args = parse_args()
    siginfo_message = 'Loading data ...'
    data, header = get_data('Data/train_set.csv')

    # train model
    folds = data.shape[0] if args['folds'] is None else args['folds']
    correct = np.zeros(folds)
    i = 0
    for train, test in KFold(n_splits=folds).split(data):
        i += 1
        siginfo_message = 'Running fold {} of {}'.format(i, folds)
        X, y = split_data(data[train, :], header)
        X_test, y_test = split_data(data[test, :], header)
        clf = RandomForestClassifier(
            n_estimators=args['estimators'],
            max_depth=args['depth'],
            class_weight='balanced_subsample')
        clf.fit(X, y)
        correct[i - 1] = clf.score(X_test, y_test)
    print("mean={0:0.4f}; sem={1:0.4f}".format(correct.mean(),
                                               st.sem(correct)))

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
