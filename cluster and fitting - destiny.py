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
#----------------------------Function defintion------------------------------#
def get_dataframes(url):
    '''
    this function cleans up a data and returns 2 dataframes. 
    1. the original data
    2. the transposed data
    
    Parameters
    ----------
    url : string
        url of the dataset to be used.

    Returns
    -------
    df : dataframe
        orignal dataframe.
    df2 : TYPE
        transposed dataframe.

    '''
    df = pd.read_excel(url, skiprows=3)
    df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
    df2 = df.T
    df2 = df2.rename(columns=df2.iloc[0])
    df2 = df2.drop(index=df2.index[0], axis=0)
    df2['Year'] = df2.index
    return df, df2

def plot_cluster(y_means, x):
    '''
    this function plots a graph of the individual clusters and their
    centeriods
    
    Parameters
    ----------
    y_means : list
        list of clusters.
    x : dataframe
        dataframe containing the clusters.

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, 
                c = 'purple',label = 'label 0')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, 
                c = 'orange',label = 'label 1')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, 
                c = 'green',label = 'label 2')
    plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, 
                c = 'blue',label = 'Iabel 3')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
                s = 10, c = 'red', label = 'Centroids')
    plt.legend()
    plt.xlabel('1975')
    plt.ylabel('2020')
    plt.title('K-means clustering’s scatter plot')
    plt.show()
    
# define the true objective function
def objective(x, a, b, c, d):
    '''
    Model to be used for our curve fitting

    '''
    x =- 2000
    return a*x**3 + b*x**2 + c*x + d

#-----------------------------Intialization-----------------------------------#
first_year = '1975'
second_year = '2020'
url = 'https://api.worldbank.org/v2/en/indicator/AG.LND.ARBL.ZS?downloadformat=excel'
df, df2 = get_dataframes(url)

cluster_data = df.loc[df.index, ["Country Name", first_year, second_year]]
print(cluster_data)
print()
#extracting of the require data for the clustering prediction
x = cluster_data[[first_year, second_year]].dropna().values

fitting_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', 
        '2017', '2018', '2019', '2020']

# ------------ clustering------------#

#visualize the original cluster
plt.figure()
cluster_data.plot(second_year, first_year, kind='scatter')
plt.xlabel('1975')
plt.ylabel('2020')
plt.title('Scatterplot')
plt.legend()
plt.show()

#find the required number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, 
                    n_init= 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('getting the cluster')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, 
                n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#plot the cluster
plot_cluster(y_kmeans, x)

#--------------------------------curve fitting--------------------------------#
#get dataset
df_fitting = df2.loc[fitting_years, 
                          ["Year", "United Kingdom"]].apply(pd.to_numeric, 
                                                            errors='coerce')

x = df_fitting.dropna().to_numpy()

# choose the input and output variables
x, y = x[:, 0], x[:, 1]

# curve fit
popt, _ = opt.curve_fit(objective, x, y)

# summarize the parameter values
a, b, c, d = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c, ))
print()
param, covar = opt.curve_fit(objective, x, y)

sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(x, objective, popt, sigma)

print(low, up)
print()

print('covar = ', covar)
print()

# plot input vs output
plt.figure()
plt.plot(x, y)
plt.xlabel('Year')
plt.ylabel('Y')
plt.title('curve fitting')
plt.legend()
plt.savefig('save1.png')
plt.show()

# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x), max(x) + 1, 1)

# calculate the output for the range
y_line = objective(x_line, a, b, c, d)

print(y_line)
print()
