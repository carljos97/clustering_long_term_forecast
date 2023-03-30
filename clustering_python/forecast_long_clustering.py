"""
######### The point of this script is to develop the k-means algorithm without machine learning libraries #########

- The dataset (dailyconsume_id1.xlsx) has 12 months of daily consume measurements (consumer_unit_id = 1), but since we had some 
    missing data, we replicated a few days of 2021
- The k-means algorithm will be used to cluster the daily data across the year
- The output is 12 arrays of centroids (off peak and peek every day of the week, off peak every day of the weekend), 
    which will be used to set forecast values across the predicted month
- The default value for the number of cluster is 12. But when it comes to peak values, it can be less because of the 
    possibility of missing data in the initialized centroids.

"""

import pandas as pd
import numpy as np
#import random as rd
from matplotlib import pyplot as plt
#import datetime

class kmeans:

    def __init__(self, num_clusters, max_iter):
        
        self.num_clusters = num_clusters
        self.max_iter = max_iter
    
    def initalize_centroids(self, X):
        
        self.num_clusters = 12
        centroid1 = X[(X[:,1] == 3)]
        centroid2 = X[(X[:,1] == 7)]
        centroid3 = X[(X[:,1] == 11)]
        centroid4 = X[(X[:,1] == 15)]
        centroid5 = X[(X[:,1] == 19)]
        centroid6 = X[(X[:,1] == 23)]
        centroid7 = X[(X[:,1] == 27)]
        centroid8 = X[(X[:,1] == 31)]
        centroid9 = X[(X[:,1] == 35)]
        centroid10 = X[(X[:,1] == 39)]
        centroid11 = X[(X[:,1] == 43)]
        centroid12 = X[(X[:,1] == 47)]
        centroids = np.concatenate([x for x in [centroid1,centroid2,centroid3,centroid4,centroid5,centroid6,
                                centroid7,centroid8,centroid9,centroid10,centroid11,centroid12] if x.size > 0], axis = 0)
        if np.shape(centroids)[0] != 12:    # If there is no data from one of the weeks, the number of clusters should be reduced
            self.num_clusters = np.shape(centroids)[0]
            
        return centroids
            
        """
        # If it was random
        idx = np.rd.permutation(X.shape[0])
        centroids = X[idx[:self.num_clusters]]
        """
        
    def compute_centroid(self, X, labels):
  
        centroids = np.zeros((self.num_clusters, X.shape[1]))
        for k in range(self.num_clusters):
            centroids[k] = np.mean(X[labels == k], axis=0)
            
        return centroids
    
    def compute_distance(self, X, centroids):

        distances = np.zeros((X.shape[0], self.num_clusters))
        
        for k in range(self.num_clusters):
            dist = np.linalg.norm(X - centroids[k], axis=1)
            distances[:,k] = np.square(dist)
            
        return distances
    
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)
    
    def fit(self, X):
        self.centroids = self.initalize_centroids(X)
        
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroid(X, self.labels)


dataset = pd.read_excel('dailyconsume_id1.xlsx') # importing the dataset
dataset_offpeak = dataset.drop(columns=['date','peak_measurements'])
dataset_peak = dataset.drop(columns=['date','offpeak_measurements'])
dataset_peak = dataset_peak.drop(dataset_peak[dataset_peak['peak_measurements'] == 0].index).reset_index(drop=True)

num_clusters=12    # default
max_iter=1000    # default

def week_day(day_of_the_week, name):
    
    df_offpeak = dataset_offpeak.drop(dataset_offpeak[(dataset_offpeak['day_of_the_week'] != day_of_the_week)].index)  
    X_offpeak = df_offpeak.iloc[:, [0, 1]].values    # creating an array
    
    kmeansmodel_offpeak = kmeans(num_clusters, max_iter)
    kmeansmodel_offpeak.fit(X_offpeak)
    centroids_offpeak = kmeansmodel_offpeak.centroids
    
    df_peak = dataset_peak.drop(dataset_peak[(dataset_peak['day_of_the_week'] != day_of_the_week)].index)  
    X_peak = df_peak.iloc[:, [0, 1]].values    # creating an array
    
    kmeansmodel_peak = kmeans(num_clusters, max_iter)
    kmeansmodel_peak.fit(X_peak)
    centroids_peak = kmeansmodel_peak.centroids

    """    
    plt.scatter(X_offpeak[:,1],X_offpeak[:,0],c='black',label='unclustered data')
    plt.xlabel('week of the year')
    plt.ylabel('off peak measurements')
    plt.show()    
    """    

    plt.scatter(X_offpeak[:, 1], X_offpeak[:, 0], c='green', s=50, cmap='viridis')
    plt.scatter(centroids_offpeak[:, 1], centroids_offpeak[:, 0], c='red', s=200, alpha=0.5)
    plt.xlabel("week_of_the_year")
    plt.title(name)
    plt.ylabel("offpeak_measurements")
    plt.show()

    plt.scatter(X_peak[:, 1], X_peak[:, 0], c='blue', s=50, cmap='viridis')
    plt.scatter(centroids_peak[:, 1], centroids_peak[:, 0], c='red', s=200, alpha=0.5)
    plt.xlabel("week_of_the_year")
    plt.title(name)
    plt.ylabel("peak_measurements")
    plt.show()      
    
    return centroids_offpeak, centroids_peak

def weekend_day(day_of_the_weekend, name):
    
    df_offpeak = dataset_offpeak.drop(dataset_offpeak[(dataset_offpeak['day_of_the_week'] != day_of_the_weekend)].index)  
    X_offpeak = df_offpeak.iloc[:, [0, 1]].values    # creating an array
    
    kmeansmodel = kmeans(num_clusters, max_iter)
    kmeansmodel.fit(X_offpeak)
    centroids_offpeak = kmeansmodel.centroids
    
    plt.scatter(X_offpeak[:, 1], X_offpeak[:, 0], c='green', s=50, cmap='viridis')
    plt.scatter(centroids_offpeak[:, 1], centroids_offpeak[:, 0], c='red', s=200, alpha=0.5)
    plt.xlabel("week_of_the_year")
    plt.title(name)
    plt.ylabel("offpeak_measurements")
    plt.show()
    
    return centroids_offpeak


centroids_offpeak_monday, centroids_peak_monday = week_day(2, 'monday')
centroids_offpeak_tuesday, centroids_peak_tuesday = week_day(3,'tuesday')
centroids_offpeak_wednesday, centroids_peak_wednesday = week_day(4, 'wednesday')
centroids_offpeak_thursday, centroids_peak_thursday = week_day(5, 'thursday')
centroids_offpeak_friday, centroids_peak_friday = week_day(6, 'friday')

centroids_offpeak_sunday = weekend_day(1, 'sunday')
centroids_offpeak_saturday = weekend_day(7, 'saturday')

"""
def forecast(year, month, day):
    week = datetime.date(year, month, day).strftime("%V")
    return week  
week_year = forecast(2020, 3, 1)
"""







