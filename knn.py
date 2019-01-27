#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:41:36 2019

@author: peterflickinger
"""


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


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
            ans = self.targets_dtrain[index]
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
    
    

    
def load_car_data():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "saftey", "classification"]
    temp = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                  header=None, names=headers, na_values="?" )
    #convert everything to a numerical value
    cleanup = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1 },
               "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1 },
               "doors": {"5more": 6 }, 
               "persons": {"more": 6},
               "lug_boot": {"big":3, "med": 2, "small": 1},
               "saftey": {"high":3, "med": 2, "low": 1},
               "classification": {"vgood":3, "good": 2, "acc": 1, "unacc": 0}}
    
    temp.replace(cleanup, inplace=True)
    data.target = temp.classification
    data.data = temp.drop("classification", axis=1)
    data.type = "Classification"
    return data

def load_automobile_data():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
    temp = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                  header=None, names=headers, na_values="?", delim_whitespace=True)
    
    #since there are only 6 missing data points lets just drop those rows
    ans = yes_or_no("Would you like to remove the missing data")
    if (ans):
        temp = temp.dropna()
    else:
        temp = temp.fillna(temp.mean())
    #the car name would be too specific to track mpg, lets use the car brand instead
    brands = temp.car_name.str.split().str[0]
    temp['car_name'] = brands
    temp = pd.get_dummies(temp, columns=["car_name"])
    
    
    data.target = temp.mpg
    data.data = temp.drop("mpg", axis=1)
    data.type = "Regression"
    return data




def load_student_data():
    temp = pd.read_csv("student-mat.csv", true_values=["yes"], false_values=["no"],
                       na_values="?", sep=";")
    
    #since there is not missing data we should be good to go. 
    #lets just convert everything over to integers
    
    temp['school'] = temp['school'].astype('category').cat.codes
    temp['sex'] = temp['sex'].astype('category').cat.codes
    temp['address'] = temp['address'].astype('category').cat.codes
    temp['famsize'] = temp['famsize'].astype('category').cat.codes
    temp['Pstatus'] = temp['Pstatus'].astype('category').cat.codes
    temp['Mjob'] = temp['Mjob'].astype('category').cat.codes
    temp['Fjob'] = temp['Fjob'].astype('category').cat.codes
    temp['reason'] = temp['Fjob'].astype('category').cat.codes
    temp['guardian'] = temp['Fjob'].astype('category').cat.codes

    
    data.target = temp.G3
    data.data = temp.drop("G3", axis=1)
    data.type = "Regression"
    return data



def knn(data, test_size, k_max):
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=test_size)
    
    
    if ans: 
        classifier = knnClassifier()
        classifier.fit(x_train, y_train)
        pred = np.array([])
        for i in range(1, k_max):
            if (data.type == "Classification"): 
                predictions = classifier.predict(x_test, i)
                error = np.mean( predictions != y_test)
            else:
               predictions = classifier.predict_regression(x_test, i)
               error = mean_squared_error( y_test, predictions)      
            pred = np.append(pred, error)
        index = np.argpartition(pred, 1)
        k = index[0] + 1
        error = pred[index[0]]
    else:
        pred = np.array([])
        for i in range(1, k_max):
            if(data.type == "Classification"):
                classifier = KNeighborsClassifier(n_neighbors=i)
            else:
                classifier = KNeighborsRegressor(n_neighbors=i)
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            
            if(data.type == "Classification"):
                error = np.mean( predictions != y_test)
            else:
                error = mean_squared_error( y_test, predictions)
            pred = np.append(pred, error)
        index = np.argpartition(pred, 1)
        k = index[0] + 1
        error = pred[index[0]]
    
    if(data.type == "Classification"):
        error = error * 100
        
    print("Smallest Error at k = " + str(k) + " : " + str(error) + "%" + " with test size: " + str(test_size) + "\n")




reply = str(input('Please select a dataset to import : \n   Car Evaluation (c)\n   Automobile MPG (a)\n   Student Preformance (s)\n')).lower().strip()
if reply[0] == 'c':
    data = load_car_data()
if reply[0] == 'a':
    data = load_automobile_data()
if reply[0] == 's':
    data = load_student_data()

k = 0
error = 0

reply = int(input('What would you like the highest k tested to be?: '))
k_max = reply

#This does not work yet...
#ans = yes_or_no("Would you like to use the handmade classifier")
ans = False


print("=============================================\nResults\n=============================================/n")
knn(data, 0.30, k_max)
knn(data, 0.25, k_max)
knn(data, 0.20, k_max)

