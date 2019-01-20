#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:03:10 2019

@author: peterflickinger
"""

from sklearn import datasets

data = datasets.load_iris().data

row1 = data[0]
row2 = data[1]

sum = 0

for i in range(len(row1)):
    diff = row1[i] - row2[i]
    sq = diff ** 2
    sum += sq
    
#### ~~ or ~~ ####################

diff = row1 - row2

sq = diff ** 2

dist = sum(sq)


s = np.sum([[1, 2, 3], [4, 5, 6])

    
