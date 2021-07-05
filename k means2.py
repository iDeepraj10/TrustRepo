from sklearn.metrics import pairwise_distances_argmin
import pandas as pd
import numpy as np

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix B.csv")
df = df.drop(df.columns[[0]], axis =1)

y = [ele for ele in range(0,100)]
X = df[df.columns[[0]]]

centers, labels = find_clusters(X, 3)
plt.scatter(y, X, c=labels,
            s=50, cmap='viridis');

plt.show()
