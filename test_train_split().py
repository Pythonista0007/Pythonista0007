#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:04:41 2021

@author: Milind Prakash

"""
# =============================================================================
# We have a dataset containing prices of used BMW cars. 
# We are going to analyze this dataset and build a prediction function that can predict a price by taking mileage and 
# age of the car as input. We will use sklearn train_test_split method to split training and testing dataset
# 
# 
# =============================================================================
import pandas as pd
import numpy as np
df=pd.read_csv('/Users/apple/Desktop/Codebasics/py-master 2/ML/6_train_test_split/carprices.csv')
df
mapping={df.columns[1]:'Age',df.columns[2]:'Price'}
df1=df.rename(columns=mapping)
df1

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(df1['Age'],df1['Price'])
plt.scatter(df1['Mileage'],df1['Price'])
X=df1[['Mileage','Age']]
Y=df1['Price']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

X_train
X_test
Y_train
Y_test

from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(X_train,Y_train)

model.predict(X_test)

Y_test

model.score(X_test,Y_test)
































from sklearn.model_selection import train_test_split

