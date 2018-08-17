""" Data Preprocessing Template """

""" import 3 main packages """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

""" load data """
dataset = pd.read_csv('Data.csv')

""" features and labels: X should be matrix (2D), y should be array (1D) """
# iloc (index): select all rows, all columns except last // values:  as np array
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

""" take care of missing data """
# specify a value as missing if it is NaN, replace with mean, do WRT columns
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:])                             # fit() to learn
X[:, 1:] = imputer.transform(X[:, 1:])                      # transform() to apply

""" encode categorical data """
# by default, LabelEncoder does not encode as 1-hot array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# need to use 1-hot, specify first column of X
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# also encode labels y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

""" train/test split """
from sklearn.model_selection import train_test_split
# keep track of order, 20% of data as test, random state to get reproducible results, defaults to shuffle=True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

""" feature scaling """
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# already fitted, can just transform
X_test = sc_X.transform(X_test)






