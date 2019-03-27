#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:32:55 2019

@author: peterflickinger
"""
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor 
from sklearn import preprocessing

import numpy as np
import pandas as pd


class Data:
    data = []
    target = []
    
    
    
temp = pd.read_csv("listings.csv", na_values="N/A", parse_dates=[3,22,75,77,78], true_values=['t'], false_values='f')

#note experiences_offered are all none
temp.drop(['listing_url', 'scrape_id', 'name', 'summary','space','description','neighborhood_overview', 'notes','transit', 'access', 'interaction','house_rules','thumbnail_url','medium_url','picture_url','xl_picture_url','host_url', 'host_picture_url','host_name','host_about','host_thumbnail_url','neighbourhood', 'state', 'market', 'smart_location', 'country_code', 'country', 'experiences_offered', 'id', 'last_scraped', 'host_id', 'host_since', 'host_neighbourhood', 'host_verifications', 'amenities', 'calendar_updated', 'calendar_last_scraped', 'first_review', 'last_review', 'license', 'jurisdiction_names', 'cancellation_policy'], axis=1, inplace=True)

#convert over all the date times
#temp.last_scraped = pd.to_datetime(temp.last_scraped)
#temp.host_since = pd.to_datetime(temp.host_since)
#temp.calendar_last_scraped = pd.to_datetime(temp.calendar_last_scraped)
#temp.first_review = pd.to_datetime(temp.first_review)
#temp.last_review = pd.to_datetime(temp.last_review)

temp['host_is_local'] = (temp.host_location.str.find('San Francisco') > -1).astype(bool)
temp.drop(['host_location'], axis=1, inplace=True)

#clean up dollar signs and percentages
temp.host_response_rate = (temp.host_response_rate.str.replace('%', '')).fillna(0).astype(int)
temp.price = temp.price.replace('[\$,]', '', regex=True).fillna(0).astype(float)
temp.weekly_price = temp.weekly_price.replace('[\$,]', '', regex=True).fillna(0).astype(float)
temp.monthly_price = temp.monthly_price.replace('[\$,]', '', regex=True).fillna(0).astype(float)
temp.security_deposit = temp.security_deposit.replace('[\$,]', '', regex=True).fillna(0).astype(float)
temp.cleaning_fee = temp.cleaning_fee.replace('[\$,]', '', regex=True).fillna(0).astype(float)
temp.extra_people = temp.extra_people.replace('[\$,]', '', regex=True).fillna(0).astype(float)


#clean up street names and cities
temp.neighbourhood_cleansed = temp.neighbourhood_cleansed.astype('category').cat.codes
temp.city = temp.city.astype('category').cat.codes
temp.property_type = temp.property_type.astype('category').cat.codes
temp.street = temp.street.astype('category').cat.codes


cleanup = {"bed_type": {"Real Bed": 4, "Futon": 3, "Pull-out Sofa": 2, "Airbed": 1, "Couch":0},
           "host_response_time": {"within an hour": 3, "within a few hours": 2, "within a day": 1, "a few days or more": 0 },
           "room_type": {"Entire home/apt": 3, "Private room": 2, "Shared room": 1}}
temp.replace(cleanup, inplace=True)

cols = temp.columns
temp[cols] = temp[cols].apply(pd.to_numeric, errors='coerce')

#pd.to_numeric(temp, errors='coerce')
temp.fillna(temp.mean, inplace=True)
#temp_norm = preprocessing.normalize(temp)

data = Data()
data.target = temp.review_scores_rating

data.data = temp.drop(['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
classifier = MLPRegressor(max_iter=700, batch_size=50) 
    
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
print("Predictions: " + str(pred))
acc = classifier.score(x_test, y_test)
print("Accuracyt: " + str(acc))
print("layers: " + str(classifier.n_layers_))


















