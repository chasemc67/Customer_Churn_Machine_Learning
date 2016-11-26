#!/usr/bin/env python

import pandas as pd


def main():
    data = pd.read_csv('Data/train_set.csv')
    majority_class = data['renewed'].mode()
    num_majority = len(data[data['renewed'] == int(majority_class)])
    print('Majority Classifier Accuracy: {0:0.4f}'.format(
        num_majority / len(data)))

if __name__ == '__main__':
    main()
