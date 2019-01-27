#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:41:36 2019

@author: peterflickinger
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
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



data = pd.read_csv("adult.csv", names=["age", "workclass", "fnlwgt", "education", "education-num",
                                       "marital-status", "occupation", "relationship", "race", "sex", 
                                       "capital-gain", "capital-loss", "hours-per-week", "native-country", 
                                       "income"])

cleanup = {"education":   {"Preschool" : 1, 
"1st-4th" : 2, 
"5th-6th" : 3, 
"7th-8th" : 4, 
"9th" : 5, 
"10th" : 6, 
"11th" : 7, 
"12th" : 8, 
"HS-grad" : 9, 
"Some-college" : 10, 
"Prof-school" : 11, 
"Assoc-voc" : 12, 
"Assoc-acdm" : 13, 
"Bachelors" : 14, 
"Masters" : 15, 
"Doctorate" : 16 }}

data.replace(cleanup, inplace=True)
data.head()



 


