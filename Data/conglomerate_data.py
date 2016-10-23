#!/usr/bin/env python

import numpy as np

filenames = ['2015-Aug.csv',
             '2015-Dec.csv',
             '2015-Jul.csv',
             '2015-Jun.csv',
             '2015-Nov.csv',
             '2015-Oct.csv',
             '2015-Sep.csv',
             '2016-Apr.csv',
             '2016-Feb.csv',
             '2016-Jan.csv',
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
    data.append(header)

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
for row in data[1:]:
    for i, entry in enumerate(row):
        if entry == '':
            row[i] = str(np.nan)
    if float(row[- 6]) == 0:
        row[- 6] = str(np.nan)
for i in range(len(data)):
    data[i] = data[i][:27] + data[i][39:]

# write data
with open('data.csv', 'w') as outfile:
    for row in data:
        outfile.write(','.join(row) + '\n')
