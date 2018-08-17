""" Simple Linear Regression model """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# import data
dataset = pd.read_csv('Salary_Data.csv')

# get features and labels as np arrays
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# fit linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict on test set
y_pred = regressor.predict(X_test)

# visualize training set results
plt.scatter(X_train, y_train, color='tomato')
plt.plot(X_train, regressor.predict(X_train), color='skyblue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualize test set results
plt.scatter(X_test, y_test, color='tomato')
plt.plot(X_train, regressor.predict(X_train), color='skyblue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

