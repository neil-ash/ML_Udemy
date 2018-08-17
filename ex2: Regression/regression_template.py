""" Regression Template """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# load data
dataset = pd.read_csv('Position_Salaries.csv')

# iloc (index): select all rows, all columns except last // values:  as np array
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# train/test split
"""from sklearn.model_selection import train_test_split
# keep track of order, 20% of data as test, random state to get reproducible results, defaults to shuffle=True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"""

# visualize data
plt.scatter(X, y, color='tomato')
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.title('Salary vs Rank')

# fit regression model to dataset
""" CREATE REGRESSOR HERE """

# predict new result with polynomial regression
y_pred = regressor.predict(6.5)

# visualize regression results
plt.plot(X, regressor.predict(X), color='skyblue')
plt.show()



