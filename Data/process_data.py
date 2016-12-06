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

# get header
with open(filenames[0], 'r') as infile:
    header = infile.readline().strip().split(',')

    # extract target
    target = header[- 1]
    header.pop()

    # add month and year to header
    header += ['year', 'month']

    # append the target
    header += [target]

    # clean the header
    header = [str(entry).strip() for entry in header]

# get data
for filename in filenames:
    with open(filename, 'r') as infile:
        rows = [row.strip().split(',') for row in infile][1:]
    for row in rows:
        # extract target
        target = row[- 1]
        row.pop()

        # add month and year to row
        row += [filename.split('-')[0],
                str(months[filename[:- 4].split('-')[1]])]

        # append the target
        row += [target]

        # clean the row
        row = [str(entry).strip() for entry in row]
    data += rows

# clean data
for row in data:
    for i, entry in enumerate(row):
        if entry == '':
            row[i] = str(np.nan)

# make temporal holdout
temporal_holdout = [
    row for row in data
    if (row[- 3] == '2016')
    and (row[- 2] in [str(months['Mar']), str(months['May'])])]
data = [row for row in data if row not in temporal_holdout]
temporal_holdout = [
    row for row in temporal_holdout
    if row[header.index('cancellation_request')].lower() == 'false']
random.shuffle(temporal_holdout)

# make random holdout
data_no_cancel = [
    row for row in data
    if row[header.index('cancellation_request')].lower() == 'false']
random_holdout = random.sample(data_no_cancel,
                               math.ceil(len(data_no_cancel) * 0.3))
random.shuffle(random_holdout)
data = [row for row in data if row not in random_holdout]

# make train set
train_set = data
random.shuffle(data)

# write data
with open('temporal_holdout.csv', 'w') as outfile:
    outfile.write(','.join(header) + '\n')
    for row in temporal_holdout:
        outfile.write(','.join(row) + '\n')
with open('random_holdout.csv', 'w') as outfile:
    outfile.write(','.join(header) + '\n')
    for row in random_holdout:
        outfile.write(','.join(row) + '\n')
with open('train_set.csv', 'w') as outfile:
    outfile.write(','.join(header) + '\n')
    for row in train_set:
        outfile.write(','.join(row) + '\n')
