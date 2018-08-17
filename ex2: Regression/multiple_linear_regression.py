""" Multivariable Linear Regression model """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# load data
dataset = pd.read_csv('50_Startups.csv')

# get features and labels as np arrays
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# by default, LabelEncoder does not encode as 1-hot array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
# need to use 1-hot, specify first column of X
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[:, 1:]

# train/test split (40/10)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

""" No need to apply feature scaling for linear regression """

# fit multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting test set results
y_pred = regressor.predict(X_test)

# building optimal model using backward elimination
import statsmodels.formula.api as sm
# add column of ones (const term) to X
X = np.hstack((np.ones((50, 1)), X))
# store statistically significant features in X
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# first elimination: x2 (highest p value)
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# second elimination: x3 (highest p value)
X_opt = X[:, [0, 1, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# third elimination: const (highest p value)
X_opt = X[:, [1, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# done with elimination! now, know to use features 1, 4, and 5 in X





