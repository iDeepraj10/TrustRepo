import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import statistics

#df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Dataset\\matrix 3.3.3.csv")
#df = df.drop(df.columns[[0]], axis =1)
#print(df)

def cen(df):
	centroid = []

	for i in range(0,100):
		x = df[df.columns[[i]]]
		x.dropna(axis = 0, how ='any', inplace = True)
		#print(x)
		kmeans = KMeans(3)
		kmeans.fit(x)

		identified_clusters = kmeans.fit_predict(x)

		m = statistics.mode(identified_clusters)

		closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, x)
	#print(closest)
		c1 = closest[m]
		x = np.array(x)
		centroid.append(int(x[c1]))

	return centroid