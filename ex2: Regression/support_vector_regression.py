""" Support Vector Regression """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

dataset = pd.read_csv('Position_Salaries.csv')

# iloc (index): select all rows, all columns except last // values:  as np array
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

""" NEED TO FEATURE SCALE WITH SVR """
from sklearn.preprocessing import StandardScaler
# need to create 2 separate scaler objects -- need to fit for each
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# fit SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# make prediction on a new point
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# visualize data and SVR predictions
plt.scatter(X, y, color='tomato')
plt.xlabel('Rank')
plt.ylabel('Salary')
plt.title('Salary vs Rank')
plt.plot(X, regressor.predict(X), color='skyblue')
plt.show()

