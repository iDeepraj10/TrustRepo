

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial import distance


df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix3.1.csv")
df1=df




df1= df1.fillna(0)
print(df1)



def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row

df_std = df1.apply(standardize)
print(df_std)



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
similarity= euclidean_distances(df_std)
print(similarity)



similarity_df = pd.DataFrame(similarity)
print(similarity_df)



def get_similar_consumer(c):
    sim_con= similarity_df[c]
    print(sim_con)
    print("similarity consumer : ")
    sim_con= sim_con.sort_values(ascending=True)
    return sim_con




x=get_similar_consumer(2)



print(x)
