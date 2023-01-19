import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.optimize as opt
import err_ranges as err

"""
preprocessing our raw data for example 
scaling the data, normalising it, managing missing numbers, 
and data cleansing and transformation.
and also reading my dataset
"""
first_year = '1975'
second_year = '2020'
df = pd.read_excel('https://api.worldbank.org/v2/en/indicator/AG.LND.ARBL.ZS?downloadformat=excel', skiprows=3)
df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
cluster_data = df.loc[df.index, ["Country Name", first_year, second_year]]

x = cluster_data[[first_year, second_year]].dropna().values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init= 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('getting the cluster')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()
 
