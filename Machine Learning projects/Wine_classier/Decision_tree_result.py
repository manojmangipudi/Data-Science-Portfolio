# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:02:34 2021

@author: manoj
"""
"""
ENTROPY: is a measure of randomness in the data.

INFOTMATION GAIN: The information gain is based on the decrease in entropy 
after a dataset is split on an attribute. Constructing a decision tree is all
about finding attribute that returns the highest information gain 
(i.e., the most homogeneous branches).

GINI IMPURITY: Measures the impurity of the nodes 
            gini_impurity = 1 - gini
            gini = (p1^2 + p2^2 + ....+ pn^2)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import sklearn
import time
import matplotlib.pyplot as plt


# Get the Data from respective place in the system 
red = pd.read_csv('C:\\Users\manoj\OneDrive\Desktop\P_kit\Project_1\Wine_quality\.spyproject\winequality-red.csv', header=0, low_memory=False, sep = ';')
white = pd.read_csv("C:\\Users\manoj\OneDrive\Desktop\P_kit\Project_1\Wine_quality\.spyproject\winequality-white.csv",header=0, low_memory=False, sep = ';')

# here we have two datasets with red wine and white wine. we write a function to call both the datasets. 
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

tic = time.process_time()

#-----------------------------Decision Tree------------------------------------

def decis_tree(wine_set):
    
    w = wine_set
    #recode quality
    recode = {3:0, 4:0, 5:0, 6:1, 7:1, 8:1, 9:1}
    wine_set['quality_c'] = wine_set['quality'].map(recode)
    
    #split data
    X = wine_set.values[:,0:11]
    y = wine_set.quality_c
    
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = .4)
    
    #MODEL BUILD
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    #Confusion Matrix
    print(sklearn.metrics.confusion_matrix(y_test, predictions))
    print(sklearn.metrics.accuracy_score(y_test, predictions))
    
    plt.figure(figsize = (25,15))
    if w.equals(red):
        # to get a plot of tree
        plot_tree(classifier, filled = True)
        # export the decision tree 
        export_graphviz(classifier, out_file="red_decision_tree.dot")
    else:
        # to get a plot of tree
        plot_tree(classifier, filled = True)
        # export the decision tree 
        export_graphviz(classifier, out_file="white_decision_tree.dot")
        
call(decis_tree)    
    
    















