# import the libraries
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
# ensemble is when we use multiple algorithm to predict the outcome
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
digits = load_digits()
    
# check the directories
print(dir(digits))

# Check the digits
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])

# dataframe of digits input
df = pd.DataFrame(digits.data)

# add target data to dataframe
df['target'] = digits.target

# train test split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis = 'columns'),digits.target, test_size = 0.2) 

# --------------model built--------------------------
# n_estimators = 100 here!!!
model = RandomForestClassifier()

# fit the model with training data
model.fit(X_train, y_train)

# check the accuracy with testing data
accuracy_Score = model.score(X_test, y_test)

# model prediction now
y_predicted = model.predict(X_test)

# ----------------Confusion Matrix--------------------

cm = confusion_matrix(y_test, y_predicted)

# lets visualise the Confusion Matrix

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True)
plt.xlabel('predicted')
plt.ylabel('Truth')
