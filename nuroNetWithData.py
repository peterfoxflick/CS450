#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:41:36 2019

@author: peterflickinger
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor 
from sklearn import preprocessing


import pandas as pd
import numpy as np
import warnings
#This will get rid of warnings on the screen about terminating early
warnings.filterwarnings('ignore', 'Stochastic Optimizer:*')

class Data:
    type = ""
    data = []
    target = []
    
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
                  header=None, names=headers, na_values="?")
    #convert everything to a numerical value
    cleanup = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1 },
               "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1 },
               "doors": {"5more": 6 }, 
               "persons": {"more": 6},
               "lug_boot": {"big":3, "med": 2, "small": 1},
               "saftey": {"high":3, "med": 2, "low": 1},
               "classification": {"vgood":3, "good": 2, "acc": 1, "unacc": 0}}
    
    temp.replace(cleanup, inplace=True)
    data = Data()
    data.target = temp.classification
    data.data = temp.drop("classification", axis=1)
    data.type = "Classification"
    runNet(data)

def load_automobile_data():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]

    
    
    
    temp = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                  header=None, names=headers, na_values="?", delim_whitespace=True)
    
    #since there are only 6 missing data points lets just drop those rows
    temp = temp.fillna(temp.mean())
    #the car name would be too specific to track mpg, lets use the car brand instead
    brands = temp.car_name.str.split().str[0]
    temp['car_name'] = brands
    temp = pd.get_dummies(temp, columns=["car_name"])
    
    data = Data()
    data.target = temp.mpg
    data.data = temp.drop("mpg", axis=1)
    data.type = "Regression"
    print("FILL MISSING DATA")
    runNet(data)
    
    
    
    temp = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                  header=None, names=headers, na_values="?", delim_whitespace=True)
    

    
    
    #since there are only 6 missing data points lets just drop those rows
    temp = temp.dropna()

    #the car name would be too specific to track mpg, lets use the car brand instead
    brands = temp.car_name.str.split().str[0]
    temp['car_name'] = brands
    temp = pd.get_dummies(temp, columns=["car_name"])
    
    data = Data()
    data.target = temp.mpg
    data.data = temp.drop("mpg", axis=1)
    data.type = "Regression"
    print("DROP MISSING DATA")
    runNet(data)




def load_student_data():
    temp = pd.read_csv("student-mat.csv", true_values=["yes"], false_values=["no"],
                       na_values="?", sep=";")
    
    #since there is not missing data we should be good to go. 
    #lets just convert everything over to integers
    
    temp = pd.get_dummies(temp)
    data = Data()
    data.target = temp.G3
    data.data = temp.drop("G3", axis=1)
    data.type = "Regression"
    
    
    print("Dummies")
    runNet(data)
        
    temp = pd.read_csv("student-mat.csv", true_values=["yes"], false_values=["no"],
                       na_values="?", sep=";")
    
    #since there is not missing data we should be good to go. 
    #lets just cleanup the data
    
    temp['school'] = temp['school'].astype('category').cat.codes
    temp['sex'] = temp['sex'].astype('category').cat.codes
    temp['address'] = temp['address'].astype('category').cat.codes
    temp['famsize'] = temp['famsize'].astype('category').cat.codes
    temp['Pstatus'] = temp['Pstatus'].astype('category').cat.codes
    temp['Mjob'] = temp['Mjob'].astype('category').cat.codes
    temp['Fjob'] = temp['Fjob'].astype('category').cat.codes
    temp['reason'] = temp['Fjob'].astype('category').cat.codes
    temp['guardian'] = temp['Fjob'].astype('category').cat.codes

    data = Data()
    data.target = temp.G3
    data.data = temp.drop("G3", axis=1)
    data.type = "Regression"
    print("type")
    runNet(data)
    
    
    print("COMBINDED")
    
        
    temp = pd.read_csv("student-mat.csv", true_values=["yes"], false_values=["no"],
                       na_values="?", sep=";")
    temp['school'] = temp['school'].astype('category').cat.codes
    temp['sex'] = temp['sex'].astype('category').cat.codes
    temp['address'] = temp['address'].astype('category').cat.codes
    temp['famsize'] = temp['famsize'].astype('category').cat.codes
    temp['Pstatus'] = temp['Pstatus'].astype('category').cat.codes
    temp = pd.get_dummies(temp)
    data = Data()
    data.target = temp.G3
    data.data = temp.drop("G3", axis=1)
    data.type = "Regression"
    
    
    
    runNet(data)
    
def load_tuned_data():
    temp = pd.read_csv("student-mat.csv", true_values=["yes"], false_values=["no"],
                       na_values="?", sep=";")
    
    #since there is not missing data we should be good to go. 
    #lets just convert everything over to integers
    
    temp = pd.get_dummies(temp)
    data = Data()
    data.target = temp.G3
    data.data = temp.drop("G3", axis=1)
    
    data.data = preprocessing.normalize(data.data)
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    
    activations = ["identity", "logistic", "tanh", "relu"]
    sovlers = ["lbfgs", "sgd", "adam"]
    for active in activations:
        for solve in sovlers: 
            accs = []
            for i in range(10):
                classifier = MLPRegressor(max_iter=1000, batch_size=50, activation=active, solver=solve) 
                classifier.fit(x_train, y_train)
                acc = classifier.score(x_test, y_test)
                accs.append(acc)
            print("Activation is " + active + " solver is " + solve)
            print("    Accuracy: " + str(np.mean(accs)))
            
    # Now to just test arround in SGD
    learning_rates = ["constant", "invscaling", "adaptive"]
    
    for rates in learning_rates:
        for i in range(5,10):
            m = i / 10.0;
            classifier = MLPRegressor(max_iter=1000, batch_size=50, learning_rate=rates, solver="sgd", momentum=m) 
            classifier.fit(x_train, y_train)
            acc = classifier.score(x_test, y_test)
            print("learning rate is " + rates + " momentum is " + str(m))
            print("    Accuracy: " + str(acc))

    
#Multiple classifications iwth only boolean data
def load_zoo_data():
    headers = ["animal", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes","venomous","fins","legs","tail","domestic","catsize ","type"]

    temp = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data", true_values=["1"], false_values=["0"],
                       names=headers)
    temp = temp.drop("animal", axis=1)
    #since there is not missing data we should be good to go. 
    #lets just convert everything over to integers
    data = Data()
    data.target = temp.type
    data.data = temp.drop("type", axis=1)
    data.type = "Classification"
    
    runNet(data)
    
def load_balloon_data():
    headers = ["yellow", "small", "stretched", "adult", "inflated"]

    temp = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small+adult-stretch.data", 
                       true_values=["YELLOW", "SMALL", "STRETCH", "ADULT", "T"], false_values=["PURPLE", "LARGE","DIP","CHILD", "F"],
                       names=headers)
    data = Data()
    data.target = temp.inflated
    data.data = temp.drop("inflated", axis=1)
    data.type = "Classification"
    
    runNet(data)


def runNet(data):
    
    if yes_or_no("Do you want to normalize data?"):
        data.data = preprocessing.normalize(data.data)
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    
    if(data.type == "Regression"):
        classifier = MLPRegressor(max_iter=700, batch_size=50) 
    else:
        classifier = MLPClassifier(max_iter=700)
    
    classifier.fit(x_train, y_train)
    #pred = classifier.predict(x_test)
    #print("Predictions: " + str(pred))
    acc = classifier.score(x_test, y_test)
    print("Accuracy: " + str(acc))
    print("layers: " + str(classifier.n_layers_))
    
    

    
reply = str(input('Please select a dataset to import : \n   Car Evaluation (c)\n   Automobile MPG (a)\n   Student Preformance (s)\n   Zoo data (z)\n   Ballon (b)\n   Tuned on student data (t)\n')).lower().strip()
if reply[0] == 'c':
    load_car_data()
if reply[0] == 'a':
    load_automobile_data()
if reply[0] == 's':
    load_student_data()
if reply[0] == 'z':
    load_zoo_data()
if reply[0] == 'b':
    load_balloon_data()
if reply[0] == 't':
    load_tuned_data()
