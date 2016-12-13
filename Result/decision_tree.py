#!/usr/bin/env python

from scipy import stats as st
from sklearn import preprocessing, tree
from sklearn.metrics import \
    accuracy_score, \
    confusion_matrix, \
    f1_score, \
    precision_score, \
    recall_score, \
    roc_auc_score
import json
import numpy as np
import pandas as pd

TARGET_NAME = 'renewed'


def clean_data(data):
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
    np.random.shuffle(matrix)
    return matrix, header


def extract_target(data, header):
    target_index = header.index(TARGET_NAME)
    y = data[:, target_index]
    X = np.delete(data, target_index, axis=1)
    return X, y


def get_data(filename):
    data = clean_data(pd.read_csv(filename))
    matrix, header = encode_data(data)
    return matrix, header


def main():
    results = dict()

    # train learner
    train_matrix, header = get_data('Data/train_set.csv')
    X_train, y_train = extract_target(train_matrix, header)
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=5)
    clf.fit(X_train, y_train)

    # test on random holdout
    test_matrix, _ = get_data('Data/random_holdout.csv')
    X_test, y_test = extract_target(test_matrix, header)
    prediction = clf.predict(X_test)
    result = dict()
    result['accuracy'] = accuracy_score(y_test, prediction)
    result['confusion_matrix'] = confusion_matrix(y_test, prediction).tolist()
    result['f1'] = f1_score(y_test, prediction)
    result['precision'] = precision_score(y_test, prediction)
    result['recall'] = recall_score(y_test, prediction)
    result['roc_auc'] = roc_auc_score(y_test, prediction)
    result['sem'] = st.sem(y_test == prediction)
    results['random_holdout'] = result

    # test on temporal holdout
    test_matrix, _ = get_data('Data/temporal_holdout.csv')
    X_test, y_test = extract_target(test_matrix, header)
    prediction = clf.predict(X_test)
    result = dict()
    result['accuracy'] = accuracy_score(y_test, prediction)
    result['confusion_matrix'] = confusion_matrix(y_test, prediction).tolist()
    result['f1'] = f1_score(y_test, prediction)
    result['precision'] = precision_score(y_test, prediction)
    result['recall'] = recall_score(y_test, prediction)
    result['roc_auc'] = roc_auc_score(y_test, prediction)
    result['sem'] = st.sem(y_test == prediction)
    results['temporal_holdout'] = result

    # dumps results
    print(json.dumps(results,
                     sort_keys=True,
                     indent=4,
                     separators=(',', ': ')))

if __name__ == '__main__':
    # setup numpy
    np.seterr(all='raise')  # don't ignore errors

    # run
    main()
