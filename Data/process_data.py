#!/usr/bin/env python

import math
import numpy as np
import random

filenames = ['2014-Dec.csv',
             '2014-Nov.csv',
             '2014-Oct.csv',
             '2014-Sep.csv',
             '2015-Apr.csv',
             '2015-Aug.csv',
             '2015-Dec.csv',
             '2015-Feb.csv',
             '2015-Jan.csv',
             '2015-Jul.csv',
             '2015-Jun.csv',
             '2015-Mar.csv',
             '2015-May.csv',
             '2015-Nov.csv',
             '2015-Oct.csv',
             '2015-Sep.csv',
             '2016-Apr.csv',
             '2016-Aug.csv',
             '2016-Feb.csv',
             '2016-Jan.csv',
             '2016-Jul.csv',
             '2016-Jun.csv',
             '2016-Mar.csv',
             '2016-May.csv']

months = dict(zip(['Jan',
                   'Feb',
                   'Mar',
                   'Apr',
                   'May',
                   'Jun',
                   'Jul',
                   'Aug',
                   'Sep',
                   'Oct',
                   'Nov',
                   'Dec'], range(1, 13)))

data = list()

# add header
with open(filenames[0], 'r') as infile:
    header = infile.readline().strip().split(',') + ['year', 'month']
    header = [entry.strip() for entry in header]

# add data
for filename in filenames:
    with open(filename, 'r') as infile:
        rows = [row.strip().split(',') for row in infile][1:]
    for row in rows:
        row += [filename.split('-')[0],
                str(months[filename[:- 4].split('-')[1]])]
        row = [entry.strip() for entry in header]
    data += rows

# clean data
for row in data:
    for i, entry in enumerate(row):
        if entry == '':
            row[i] = str(np.nan)
    if float(row[- 6]) == 0:
        row[- 6] = str(np.nan)
for i in range(len(data)):
    data[i] = data[i][:27] + data[i][39:]

# make temporal holdout
temporal_holdout = [row for row in data
                    if (str(row[- 2]) == '2016') and
                       ((str(row[- 1]) == str(months['Mar'])) or
                        (str(row[- 1]) == str(months['May'])))]
data = [row for row in data if row not in temporal_holdout]
temporal_holdout = [header] + temporal_holdout

# make random holdout
random_holdout = random.sample(data, math.ceil(len(data) * 0.2))
data = [row for row in data if row not in random_holdout]
random_holdout = [header] + random_holdout

# make train set
train_set = [header] + data

# write data
with open('temporal_holdout.csv', 'w') as outfile:
    for row in temporal_holdout:
        outfile.write(','.join(row) + '\n')
with open('random_holdout.csv', 'w') as outfile:
    for row in random_holdout:
        outfile.write(','.join(row) + '\n')
with open('train_set.csv', 'w') as outfile:
    for row in train_set:
        outfile.write(','.join(row) + '\n')