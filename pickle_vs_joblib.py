#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:24:40 2021

@author: Milind Prakash
"""

## Predicting Home prices in Patna, Bihar
# =============================================================================
# Below is the table containing home prices in patna, Bihar.
# Here price depends on area (square feet), bed rooms and age of the home (in years). 
# Given these prices we have to predict prices of new homes based on area, bed rooms and age
# =============================================================================

# =============================================================================
# Given these home prices find out price of a home that has,
# Problem statement: Predict the price
# 3000 sqr ft area, 3 bedrooms, 40 year old
# 
# 2500 sqr ft area, 4 bedrooms, 5 year old
# 
# We will use regression with multiple variables here. Price can be calculated using following equation,
# 
# 
# Here area, bedrooms, age are called independant variables or **features** 
# whereas price is a dependant variable or Target variable
# =============================================================================

import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.read_csv('/Users/apple/Desktop/Codebasics/py-master 2/ML/2_linear_reg_multivariate/homeprices.csv')
df
mapping={df.columns[0]:'Area',df.columns[1]:'Bedrooms',df.columns[2]:'Age',df.columns[3]:'Price'}
df1=df.rename(columns=mapping)

df1

mean_rooms=df1.Bedrooms.median()
mean_rooms

df1.Bedrooms.fillna(mean_rooms,inplace=True)

df1

### ********** Model Impletation ******** #######

regression=linear_model.LinearRegression()

regression.fit(df1[['Area','Bedrooms','Age']],df1['Price'])

regression.predict([[3000,3,40]])
regression.predict([[2500,4,5]])

import pickle
with open('model_pickle','wb') as file:
    pickle.dump(regression,file)

with open('model_pickle','rb') as f:
    mp=pickle.load(f)

mp.predict([[2500,4,5]])
mp.predict([[2500,3,5]])

## Difference between Joblib & Pickle ###
# if you have large numpy array then 
##Joblib will be more efficient as compared to Pickle


import joblib
joblib.dump(regression,'model_joblib')

mj=joblib.load('model_joblib')

mj.predict([[2500,4,5]])


