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

def clean_data(data):
    data = data.drop('cancellation_request', axis=1)
    data = data.drop('id', axis=1)

    # Categorize NPS score
    mapDict = {"det": -1, "pas": 0, "prom": 1}
    for i in range(data.shape[0]):
        if np.isnan(data.xs(i)["last_nps_score"]):
            #print(data.xs(i))
            data.set_value(i, 'last_nps_score', mapDict["pas"])
        elif data.xs(i)["last_nps_score"] <= 6:
            data.set_value(i, 'last_nps_score', mapDict["det"])
        elif data.xs(i)["last_nps_score"] >= 6:
            data.set_value(i, 'last_nps_score', mapDict["prom"])
        else:
            data.set_value(i, 'last_nps_score', mapDict["pas"])
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



data, header, encoders = get_data('./train_set.csv')

print(data.xs(10))