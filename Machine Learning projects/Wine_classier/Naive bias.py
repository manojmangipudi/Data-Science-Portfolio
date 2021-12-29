# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:54:43 2021

@author: manoj
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# import data
red = pd.read_csv('C:\\Users\manoj\OneDrive\Desktop\P_kit\Project_1\Wine_quality\.spyproject\winequality-red.csv', header=0, low_memory=False, sep = ';')
white = pd.read_csv("C:\\Users\manoj\OneDrive\Desktop\P_kit\Project_1\Wine_quality\.spyproject\winequality-white.csv",header=0, low_memory=False, sep = ';')

# call the red and white wine datasets together
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
# ____________________________________Naive Bayes________________

def naive(wine_set):
   
    # recode quality (response variable) into 2 groups: 0:{3,4,5}, 1:{6,7,8,9}
    recode = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
    wine_set['quality_c'] = wine_set['quality'].map(recode)

    # split into training and testing sets
    predictors = wine_set[["density", 'alcohol', 'sulphates', 'pH', 'volatile_acidity', 'chlorides', 'fixed_acidity',
                           'citric_acid', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide']]
    targets = wine_set.quality_c

    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)
    

    classifier = GaussianNB()
    classifier.fit(pred_train, tar_train)
    predictionsG = classifier.predict(pred_test)
    # print the confusion matrix and accuracy of the model
    print(sklearn.metrics.confusion_matrix(tar_test, predictionsG))
    print(sklearn.metrics.accuracy_score(tar_test, predictionsG))
    print("Gaus " + str(classifier.score(pred_test, tar_test)))
    mse = mean_squared_error(predictionsG, tar_test)
    print(mse ** 0.5)

    classifierm = MultinomialNB()
    classifierm.fit(pred_train, tar_train)
    predictionsM = classifierm.predict(pred_test)
    # print the confusion matrix and accuracy of the model
    print(sklearn.metrics.confusion_matrix(tar_test, predictionsM))
    print(sklearn.metrics.accuracy_score(tar_test, predictionsM))
    print("Multi " + str(classifierm.score(pred_test, tar_test)))
    mse = mean_squared_error(predictionsM, tar_test)
    print(mse ** 0.5)

    classifierb = BernoulliNB()
    classifierb.fit(pred_train, tar_train)
    predictionsB = classifierb.predict(pred_test)
    # print the confusion matrix and accuracy of the model
    print(sklearn.metrics.confusion_matrix(tar_test, predictionsB))
    print(sklearn.metrics.accuracy_score(tar_test, predictionsB))
    print("Bernoulli " + str(classifierb.score(pred_test, tar_test)))
    mse = mean_squared_error(predictionsB, tar_test)
    print (mse ** 0.5)
    

print('----------------Naive Bayes------------------------')
call(naive)
