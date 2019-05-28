#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:49:05 2019

@author: molly
"""
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
# Importing the dataset
#predict how many installs will be

dataset = pd.read_csv('googleplaystore.csv')
##dataset ["Last Updated"] = pd.to_numeric(dataset ["Last Updated"])
dataset["days"]=(pd.to_datetime(dataset["Last Updated"])-pd.datetime(1970,1,1))
dataset["days"]=dataset["days"].dt.days.astype(int) 
#dataset["Category"].isna().sum()
dataset["Rating"].isna().sum()
dataset["Rating"].replace(" " ,np.NaN)
dataset["Rating"].max()
#dataset["Category"].unique()
dataset["Installs"]=dataset["Installs"].str.replace(',', '').str.replace('+', '').astype(int)
dataset["Price"]=dataset["Price"].str.replace('$', '')
dataset.loc[dataset.Type == "Free", 'Paid'] = 0 
dataset.loc[dataset.Type != "Free", 'Paid'] = 1 
dataset.loc[dataset.Rating >= 4.0, 'Good'] = 1 
dataset.loc[dataset.Rating <4.0, 'Good'] = 0 


#Reviews, Installs, Price,days
X = dataset.iloc[:, [14,3,5,13]].values
Y = dataset.iloc[:,-1].values
# Taking care of missing data

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy="median",axis=0)
imputer=imputer.fit(X[:,[0]])
imputer=imputer.fit(Y.reshape(-1, 1))
X[:,[0]]=imputer.transform(X[:,[0]]).astype(int)
Y=imputer.transform(Y.reshape(-1, 1))

# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/5, random_state=0)

'''
# Splitting the dataset into the Training set and Test set 2

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/6, random_state=0)

'''
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)
X_test = sc_X.transform(X_test)
'''
y_test_original=y_test
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_val = sc_y.transform(y_val.reshape(-1, 1))
y_test = sc_y.transform(y_test.reshape(-1, 1))
'''
'''
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
'''
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
regressor1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
regressor1.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
regressor2 = GaussianNB()
regressor2.fit(X_train, y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
regressor3 = SVC(kernel = 'rbf', random_state = 0)
#classifier = SVC(kernel = 'linear', random_state = 0)
regressor3.fit(X_train, y_train)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
regressor4 = LogisticRegression(random_state = 0)
regressor4.fit(X_train, y_train)

# Fitting Random forest to the Training set
from sklearn.ensemble import RandomForestClassifier
regressor5 = RandomForestClassifier(n_estimators = 10, random_state = 0)
regressor5.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = regressor1.predict(X_test)
y_pred2 = regressor2.predict(X_test)
y_pred3 = regressor3.predict(X_test)
y_pred4 = regressor4.predict(X_test)
y_pred5 = regressor5.predict(X_test)

'''
# Visualising the Training set results
plt.scatter(X_train[:,2], y_train, color = 'red')
plt.plot(X_train[:,2], regressor.predict(X_train), color = 'blue')
plt.title('Ratings vs Reviews (Training set)')
plt.xlabel('Reviews')
plt.ylabel('Good?')
plt.show()
# Visualising the Val set results
plt.scatter(X_val[:,2], y_val, color = 'red')
plt.plot(X_val[:,2], regressor.predict(X_val), color = 'blue')
plt.title('Ratings vs Reviews (Training set)')
plt.xlabel('Reviews')
plt.ylabel('Good?')
plt.show()
# Visualising the Test set results
plt.scatter(X_test[:,2], y_test, color = 'red')
plt.plot(X_test[:,2], regressor.predict(X_test), color = 'blue')
plt.title('Ratings vs Reviews (Training set)')
plt.xlabel('Reviews')
plt.ylabel('Good?')
plt.show()
'''

s_DT=accuracy_score(y_test, y_pred1)
s_NB=accuracy_score(y_test, y_pred2)
s_SVC=accuracy_score(y_test, y_pred3)
s_LR=accuracy_score(y_test, y_pred4)
s_RF=accuracy_score(y_test, y_pred5)

print(s_DT,s_NB,s_SVC,s_LR,s_RF)

from sklearn.metrics import average_precision_score
p_DT=average_precision_score(y_test, y_pred1)
p_NB=average_precision_score(y_test, y_pred2)
p_SVC=average_precision_score(y_test, y_pred3)
p_LR=average_precision_score(y_test, y_pred4)
p_RF=average_precision_score(y_test, y_pred5)
print(p_DT,p_NB,p_SVC,p_LR,p_RF)
#kmeans_clustering

X1=X[:,[2,3]]
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X1)

# Visualising the clusters
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Ratings')
plt.xlabel('Reviews')
plt.ylabel('Installs')
plt.legend()
plt.show()

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X1, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Reviews')
plt.ylabel('Installs')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X1)

# Visualising the clusters
plt.scatter(X1[y_hc == 0, 0], X1[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X1[y_hc == 1, 0], X1[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
#plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of Ratings')
plt.xlabel('Reviews')
plt.ylabel('Installs')
plt.legend()
plt.show()