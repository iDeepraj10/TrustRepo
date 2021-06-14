###############################################################
#####RATING PREDICTION IN THE CONTEXT OF SERVICE DISCOVERY#####
###############################################################
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from numpy import dot
from numpy.linalg import norm
#function returns medium value of a service
def median(service):
    median_values = df1.median()
    return median_values[service]

def rating(customer,service):
    x = np.array(df1)
    return x[customer][service]


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
#print(df)

df1 = np.array(df)

#Customers with NaN values along with services
M_ratings = np.argwhere(np.isnan(np.array(df)))
#print(M_ratings)

#get similarity for two customers
def similarity(c1,c2):
    temp_set = []
    count = -1
    for i,j in zip(df1[3],df1[2]): 
        count+=1  
        if np.isnan(i) or np.isnan(j):
            continue
        temp_set.append(count)

    cmp_set1 = []
    cmp_set2 = []
    for i in temp_set:
        cmp_set1.append(df1[3][i])    
        cmp_set2.append(df1[2][i])

    cos_sim = dot(cmp_set1,cmp_set2)/(norm(cmp_set1)*norm(cmp_set2))
    print(cos_sim)  



     
#print(pd.DataFrame(new_df))


