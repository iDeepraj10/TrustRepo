import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
 
#df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Dataset\\matrix 3.3.3.csv")
#df = df.drop(df.columns[[0]], axis =1)
#print(df)

def cen(df):
	centroid = []
	print("Getting centroid values for all items !!!")
	for i in range(0,317):
		x = df[df.columns[[i]]]
		x=x.dropna()
		#print(x)
		kmeans = KMeans(3)
		kmeans.fit(x)

		identified_clusters = kmeans.fit_predict(x)
		#print(identified_clusters)
		m = Counter(identified_clusters).most_common()[0][0]

		closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, x)
	#print(closest)
		c1 = closest[m]
		x = np.array(x)
		centroid.append(int(x[c1]))

	return centroid

	