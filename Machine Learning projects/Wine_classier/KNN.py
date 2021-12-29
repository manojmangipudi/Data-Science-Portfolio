# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:58:15 2021

@author: manoj
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


red = pd.read_csv('C:\\Users\manoj\OneDrive\Desktop\P_kit\Project_1\Wine_quality\.spyproject\winequality-red.csv', header=0, low_memory=False, sep = ';')
white = pd.read_csv("C:\\Users\manoj\OneDrive\Desktop\P_kit\Project_1\Wine_quality\.spyproject\winequality-white.csv",header=0, low_memory=False, sep = ';')

def call(functionToCall):
    #print('Red')
    functionToCall(red)
    
    print('\n')
    
    #print('white')
    functionToCall(white)
    print('\n')
    
# ----- to remove all spaces from column names ---------
def remove_col_spaces(wine_set):
    wine_set.columns = [x.strip().replace(' ', '_') for x in wine_set.columns]
    return wine_set

call(remove_col_spaces)

def knn(wine_set):
    
    # recode quality (response variable) into 2 groups: 0:{3,4,5}, 1:{6,7,8,9}
    recode = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
    wine_set['quality_c'] = wine_set['quality'].map(recode)

    # split into training and testing sets
    predictors = wine_set[["residual_sugar", 'alcohol']]
    targets = wine_set.quality_c

    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)
    
    
    # build model on training data
    classifier = KNeighborsClassifier()
    classifier = classifier.fit(pred_train, tar_train)

    predictions = classifier.predict(pred_test)

    # print the confusion matrix and accuracy of the model
    print(sklearn.metrics.confusion_matrix(tar_test, predictions))
    print(sklearn.metrics.accuracy_score(tar_test, predictions))
    
    print('Score:', classifier.score(pred_test, tar_test))
    print('RMSE:', mean_squared_error(predictions, tar_test) ** 0.5)

    
print('----------------KNN------------------------')
call(knn)
