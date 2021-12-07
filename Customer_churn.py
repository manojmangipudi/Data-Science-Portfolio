# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:09:23 2021

@author: manoj
"""
##------------------- import required libraries------------------------##

import pandas as pd
import numpy as np
import tensorflow as tf

##-------------------------DATA PREPROCESSING--------------------------##

mydata= pd.read_csv('Churn_Modelling.csv')

X = mydata.iloc[:, 3:-1].values
y = mydata.iloc[:,-1].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# geography germany, spain, france one hot encoding
#splitting ger, fra, spa into three diff columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#splitting the data set into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
#Feature Scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## --------------------------BUILDING ANN---------------------------##

#initializing the ANN
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

#Adding second layer
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

#Adding output layer
ann.add(tf.keras.layers.Dense(units = 1, activation= 'sigmoid'))

##-------------------------Compiling the ANN ----------------------##

ann.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##------------------training the ANN on training set---------------##

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

##--------------------------Prediction------------------------------##
"""
#-------------Predict by taking the following input-----------#
Geography: France
NOTE: 
    Here after one-hot-encoding 
    france: first_coloumn 
    spain: second_coloumn
    Germany: third_coloumn
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: \$ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: \$ 50000
"""

test_new = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 6000, 2, 1, 1, 50000]])) > 0.5

##-------------------Predicting test results--------------------##

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

pred_result = np.concatenate(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

## -------------------Confusion Matrix----------------------## 

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

accuracy_scorr = accuracy_score(y_test, y_pred)











