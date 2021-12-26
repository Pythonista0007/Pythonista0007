#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 22:29:46 2021

@author: Milind Prakash
"""

# =============================================================================
# 
# This file has car sell prices for 3 different models. First plot data points on a scatter plot chart
# to see if linear regression model can be applied. If yes, then build a model that can answer
# following questions,
# 
# **1) Predict price of a mercedez benz that is 4 yr old with mileage 45000**
# 
# **2) Predict price of a BMW X5 that is 7 yr old with mileage 86000**
# 
# **3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())**
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('/Users/apple/Desktop/Codebasics/py-master 2/ML/5_one_hot_encoding/Exercise/carprices.csv')
df
plt.scatter(df['Mileage'],df['Sell Price($)'])
plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
df.size
df.shape
from sklearn.linear_model import LinearRegression
model=LinearRegression()
mapping={df.columns[3]:'Age',df.columns[0]:'Car_Model',df.columns[1]:'Mileage',df.columns[2]:'Sell_price'}
df1=df.rename(columns=mapping)
df1
dummies=pd.get_dummies(df1['Car_Model'])
dummies
df2=pd.concat([df1,dummies],axis='columns')
df2
df2.to_csv('/Users/apple/Desktop/test07.csv')

X=df2.drop(['Car_Model','Mercedez Benz C class','Sell_price'],axis='columns')
X
Y=df2.Sell_price
Y
model.fit(X,Y)
model.predict([[45000,4,0,0]])
model.predict([[86000,7,0,1]])

model.score(X,Y)


















