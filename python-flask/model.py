# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:21:48 2020

@author: ALEKHYA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,mean_squared_error

dataset = pd.read_csv('C:/Users/ALEKHYA/Desktop/personalFolder/edu/flask/parkinsons.csv')

pkdata1=dataset

#removing outliers
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

#removing outliers



dataset=pkdata1
#
#
features=dataset.drop(['status','name'],axis=1)
labels=dataset['status']
#
#
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,mean_squared_error

scaler=MinMaxScaler((-1,1))
x=features
y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier



#model=RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
model=ExtraTreesClassifier()

model.fit(x_train,y_train)

y_pred_rf=model.predict(x_test)
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print("predict",model.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23]]))
