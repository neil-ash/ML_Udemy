""" Polynomial Regression """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# load data
dataset = pd.read_csv('Position_Salaries.csv')

# iloc (index): select all rows, all columns except last // values:  as np array
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# visualize data
plt.scatter(X, y, color='tomato')
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.title('Salary vs Rank')

# fit linear model to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# fit polynomial model to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# visualize linear regression and polynomial regression results
plt.plot(X, lin_reg.predict(X), color='navy')
plt.plot(X, lin_reg2.predict(X_poly), color='skyblue')
plt.show()

# predict new result with linear regression
lin_reg.predict(6.5)

# predict new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))

