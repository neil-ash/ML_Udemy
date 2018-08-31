""" K-Means Clustering """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

dataset = pd.read_csv('Mall_Customers.csv')

# consider annual income and spending score
X = dataset.iloc[:, [3, 4]].values

# visualize data
plt.scatter(X[:, 0], X[:, 1], color='black')
plt.title('Customer Data')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1 - 100)')

# need to determine k: use elbow method
from sklearn.cluster import KMeans

# create list of squared distance errors, check k = 1 -> 10
wcss = []
for i in range(1, 11):
    # create object to perform clustering w/ given k, use smart initialization, default num iterations, num different initializations
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plot elbow method graph
k = [i for i in range(1, 11)]
plt.plot(k, wcss, color='black')
plt.title('Elbow Method: WCSS vs k')
plt.xlabel('k')
plt.ylabel('WCSS')

# apply k-means w/ best k value (5)
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)



