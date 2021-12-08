# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:31:54 2021

@author: manoj
"""

#Module Imports
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

random.seed(101)

# set image height and width

mnist_image_height = 28
mnist_image_width = 28

#Import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data
x_train = x_train/255
x_test = x_test/255

#flatten the data
X_train_flatten = x_train.reshape(len(x_train), 28*28)
X_test_flatten = x_test.reshape(len(x_test), 28*28)

from tensorflow import keras
#model
"""
Here we use two layers one input layer and other hidden layer,
more layers we use more accuracy of prediction.
"""
model1 = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(10, activation= 'sigmoid')
    ])

#compile
compile1  = model1.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )

model1.fit(X_train_flatten, y_train, epochs = 5)

#evaluate
evaluate_test = model1.evaluate(X_test_flatten, y_test)

# predicted
y_pred = model1.predict(X_test_flatten)

#we get max of the values predicted
y_predicted_labels = [np.argmax(i) for i in y_pred]

#Build Confusion Matrix
cm = tf.math.confusion_matrix(labels = y_test, predictions= y_predicted_labels)

#visualization of Confusion Matrix
import seaborn as sn
plt.figure(figsize  = (10,7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')



