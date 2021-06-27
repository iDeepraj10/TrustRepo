###############################################################
#####RATING PREDICTION IN THE CONTEXT OF SERVICE DISCOVERY#####
###############################################################
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial import distance
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from numpy import dot
from numpy.linalg import norm
import csv


#function returns medium value of a service
def median(service):
    median_values = df.median()
    x = median_values[service]
    return x

def rating(customer,service):
    return df1[int(customer)][int(service)]


#function returns weight of a customer for specific service
def loc_weight(c,s):
    Central_point = median(s)
    #print(Central_point)
    Rating = rating(c,s)
    Weight = 1 - abs((Central_point - Rating))/10
    #print(Weight," ",Central_point," ",Rating," of :",c," ",s)
    return Weight

def glo_weight(c):
    mean1 = np.nanmedian(df1,axis=1)
    w = mean1[int(c)]
    return w

def predict(C,S):
    W_loc = loc_weight(C,S)
    W_glob = glo_weight(C)
    W_glob = W_glob/10
    R = rating(C,S)
    M_rate = (k*W_glob)+((1-k)*W_loc) * R 
    print(W_loc," ",W_glob," ",M_rate)
    print("****************************")
    return M_rate

df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix3.1.csv")


df1 = np.array(df)
k = 0.1

#Customers with NaN values along with services
M_ratings = np.argwhere(np.isnan(np.array(df)))
print(M_ratings)

#get similarity for two customers
def similarity(c1,c2):
    temp_set = []
    count = -1
    for i,j in zip(df1[c1],df1[c2]): 
        count+=1  
        if np.isnan(i) or np.isnan(j):
            continue
        temp_set.append(count)

    cmp_set1 = []
    cmp_set2 = []
    for i in temp_set:
        cmp_set1.append(df1[c1][i])    
        cmp_set2.append(df1[c2][i])

    cos_sim = np.corrcoef(cmp_set1, cmp_set2)
    return cos_sim[0,1]  
res = 0
sum1 = 0
for rate in M_ratings:
    sim_mat = {}
    for cus in df1:
        x = similarity(rate[0],int(cus[0]))
        sim_mat.update({cus[0] : x })
    sim_cus =  dict(sorted(sim_mat.items(), key=lambda item: item[1]))
    count = 0
    for i in sim_cus:
            res = predict(i,rate[1])
            if np.isnan(res):
                continue   
            sum1 = sum1 + res
            count+=1     
            #print('Rate of ',i,' is :',sum1," and count is : ",count)
            if count >= 10:
                break
    sum1 = sum1/10
    sum1 = round(sum1,2)
    df1[rate[0]][rate[1]] = sum1
    print(rate[0]," ",rate[1]," ",sum1)
    print("----------------------------------------------------")


print(pd.DataFrame(df1))
pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_data3.1.csv")
