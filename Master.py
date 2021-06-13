###############################################################
#####RATING PREDICTION IN THE CONTEXT OF SERVICE DISCOVERY#####
###############################################################
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

#function returns medium value of a service
def median(service):
    median_values = df1.median()
    return median_values[service]

def rating(customer,service):
    x = np.array(df1)
    return x[customer][service]

def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row    

def get_similar_consumer(c):
    sim_con= similarity_df[c]
    print(sim_con)
    print(c)
    sim_con= sim_con.sort_values(axis=c,ascending=True)
    return sim_con.index[1]


#function returns weight of a customer for specific service
def weight(c,s):
    Central_point = median(s)
    Rating = rating(c,s)
    Weight = (1 - abs((Central_point - Rating))/10)
    return Weight

def predict(C,S):
    W = weight(C,S)
    R = rating(C,S)
    M_rate = W * R 
    return M_rate

df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix3.1.csv")
df1=df
print(df)

df1= df1.fillna(0)
#print(df1)


#Customers with NaN values along with services
M_ratings = np.argwhere(np.isnan(np.array(df)))
print(M_ratings)


#standardize the dataset 

df_std = df1.apply(standardize)
#print(df_std)

#find similarity matrix
similarity= euclidean_distances(df_std)
#print(similarity)


similarity_df = pd.DataFrame(similarity)
#print(similarity_df)

new_df = np.array(df1)

for rate in M_ratings:
        temp = get_similar_consumer(rate[0])
        x = predict(temp,rate[1])       
        new_df[rate[0]][rate[1]] = x

#print(pd.DataFrame(new_df))
