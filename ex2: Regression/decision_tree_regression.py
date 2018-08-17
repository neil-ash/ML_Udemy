""" Decision Tree Regression model """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

dataset = pd.read_csv('Position_Salaries.csv')

# iloc (index): select all rows, all columns except last // values:  as np array
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# fit decision tree regressor to data
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# predict new result
y_pred = regressor.predict(6.5)

# visualize data
plt.scatter(X, y, color='tomato')
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.title('Salary vs Rank')

# visualize decision tree predictions
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
plt.plot(X_grid, regressor.predict(X_grid), color='skyblue')
plt.show()
