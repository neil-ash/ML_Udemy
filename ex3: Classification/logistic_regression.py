""" Logistic Regression """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

dataset = pd.read_csv('Social_Network_Ads.csv')

# only use age and salary
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# train/test: 300/100
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# need to feature scale: age and salary are on very different scales
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fit logistic regression to train set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# predict test set results
y_pred = classifier.predict(X_test)

# create confusion matrix of results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# visualize training set results
plt.title('Decision Boundary on Training Set')
plt.xlabel('Age')
plt.ylabel('Salary')
for i in range(y_train.size):
    if y_train[i] == 1:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='green')
    elif y_train[i] == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='tomato')
# get parameters
theta0 = classifier.intercept_[0]
theta1 = classifier.coef_[0, 0]
theta2 = classifier.coef_[0, 1]
# get X, y pts to plot
X_pt = min(X_train[:, 0]), \
       max(X_train[:, 0])
y_pt = (-theta0 - theta1 * X_pt[0]) / theta2, \
       (-theta0 - theta1 * X_pt[1]) / theta2
# plot decision boundary in black
plt.plot(X_pt, y_pt, color='black')

# visualize test set results
plt.title('Decision Boundary on Test Set')
plt.xlabel('Age')
plt.ylabel('Salary')
for i in range(y_test.size):
    if y_test[i] == 1:
        plt.scatter(X_test[i, 0], X_test[i, 1], color='green')
    elif y_test[i] == 0:
        plt.scatter(X_test[i, 0], X_test[i, 1], color='tomato')
# get parameters
theta0 = classifier.intercept_[0]
theta1 = classifier.coef_[0, 0]
theta2 = classifier.coef_[0, 1]
# get X, y pts to plot
X_pt = min(X_test[:, 0]), \
       max(X_test[:, 0])
y_pt = (-theta0 - theta1 * X_pt[0]) / theta2, \
       (-theta0 - theta1 * X_pt[1]) / theta2
# plot decision boundary in black
plt.plot(X_pt, y_pt, color='black')




