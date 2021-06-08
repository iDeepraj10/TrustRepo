

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial import distance


df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix3.1.csv")
df1=df




df1= df1.fillna(0)
#print(df1)


#
#def standardize(row):
#    new_row = (row - row.mean())/(row.max()-row.min())
#    return new_row

#df_std = df1.apply(standardize)
#print(df_std)



#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import euclidean_distances
#similarity= euclidean_distances(df_std)
#print(similarity)



#similarity_df = pd.DataFrame(similarity)
#print(similarity_df)



#def get_similar_consumer(c):
#    sim_con= similarity_df[c]
#    print(sim_con)
#    print("similarity consumer : ")
#    sim_con= sim_con.sort_values(ascending=True)
#    return sim_con








































#function returns medium value of a service
def median(service):
    median_values = df1.median()
    return median_values[service]

def rating(customer,service):
    x = np.array(df)
    return x[custome][service]
   

#Customers with NaN values along with services
x = np.argwhere(np.isnan(np.array(df)))


#function returns weight of a customer for specific service
def weight(c,s):
    Central_point = median(service)
    Rating = rating(c,s)
    Weight = (1 - abs((Central_point - Rating))/10)
    return Weight



















