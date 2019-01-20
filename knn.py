#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:41:36 2019

@author: peterflickinger
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class knnClassifier:

    
    def fit(self, data_train, targets_train):
        self.data_train = data_train
        self.targets_train = targets_train
        return 0
   
    def predict(self, data_test, k):
        pred = np.array([])

        for i in range(len(data_test)):
            diff = self.data_train - data_test[i]
            sq = diff ** 2
            dis = np.sum(sq, 1)
            index = np.argpartition(dis, k)
            index = index[:k]
            ans = self.targets_train[index]
            r = np.bincount(ans).argmax()
            pred = np.append(pred, r)
        return pred
    
    
    def predict_regression(self, data_test, k):
        pred = np.array([])

        for i in range(len(data_test)):
            diff = self.data_train - data_test[i]
            sq = diff ** 2
            dis = np.sum(sq, 1)
            index = np.argpartition(dis, k)
            index = index[:k]
            ans = self.targets_train[index]
            r = np.mean(ans)
            pred = np.append(pred, r)
        return pred
    
        
# thanks https://gist.github.com/garrettdreyfus/8153571
def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Sorry, please enter a proper answer (y or n)")




data = datasets.load_iris()


k = 0
error = 0
ans = yes_or_no("Would you like to use the handmade classifier")


if ans: 
    ans = yes_or_no("Would you like to use the boston housing data")
    if ans:
        data = datasets.load_boston()
        data.name = "BOSTON"
    else: 
        data = datasets.load_iris()
        data.name = "IRIS"
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30)

    classifier = knnClassifier()
    classifier.fit(x_train, y_train)
    pred = np.array([])
    for i in range(1, 7):
        if (data.name == "BOSTON"): 
            predictions = classifier.predict_regression(x_test, i)
            error = np.mean( (predictions - y_test) ** 2 > 4 ) 
        else: 
            predictions = classifier.predict(x_test, i)
            error = np.mean( predictions != y_test)
        pred = np.append(pred, error)
    index = np.argpartition(pred, 1)
    k = index[0] + 1
    error = pred[index[0]]
else:
    data = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30)
    pred = np.array([])
    for i in range(1, 7):
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        predictions = classifier.predict(x_test)
        error = np.mean( predictions != y_test)
        pred = np.append(pred, error)
    index = np.argpartition(pred, 1)
    k = index[0] + 1
    error = pred[index[0]]

print("Smallest Error at k = " + str(k) + " : " + str(error * 100) + "%")