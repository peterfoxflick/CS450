#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:47:54 2019
Boiler Plate, load and sort data
@author: peterflickinger
"""

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np


class HardCodedClassifier:
    def fit(self, data_train, targets_train):
       return 0
   
    def predict(self, data_test):
        pred = np.array([])
        for d in data_test:
            #print(d)
            pred = np.append(pred, 0)
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



iris = datasets.load_iris()

#iris_all = np.array(iris.data)
#iris_answer = np.array([np.array(iris.target)])
#iris_answers = np.concatenate((iris_all, iris_answer), 1)

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(np.array([np.array(iris.target)]))

# Show the actual target names that correspond to each number
#print(iris.target_names)

#Used to test if the answers are poperly shuffled
#iris.all = np.concatenate((iris.data, np.array([np.array(iris.target)]).T), axis=1)
#print(iris.all)


x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30)


#Test to ensure the answers were properly shuffeld
#iris.train = np.concatenate((x_train, np.array([y_train]).T), axis=1)
#found = any((iris.all[:]==iris.train[30]).all(1))

ans = yes_or_no("Would you like to use the Guasian classifier")
if ans: 
    classifier = GaussianNB()
else:
    classifier = HardCodedClassifier()


#classifier = HardCodedClassifier()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
error = np.mean( predictions != y_test)
print("Error: " + str(error))