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
import math


#function returns medium value of a service
def median(service):
    median_values = df.median()
    x = median_values[service]
    return x

def rating(customer,service):
    #print(customer," ",service)
    return df1[int(customer)][int(service)]


#function returns weight of a customer for specific service
def loc_weight(c,s):
    Central_point = median(s)
    #print(Central_point)
    Rating = rating(c,s)
    Weight = 1 - abs((Central_point - Rating))/10
    #print(Weight," ",Central_point," ",Rating," of :",c," ",s)
    return Weight

def glob_weight():
    sum1 = 0
    count = 0
    for c1 in range(0,100):
        for cus in df1:
            wl = loc_weight(c1,int(cus[0]+1))
            if math.isnan(wl):
                wl = 0
                continue
            #print("loc weight for ",cus[0]," of ",c1," is ",wl)
            sum1 = sum1 + wl
            count = count + 1
            res = sum1/count
        g_wg.update({c1 : res})    
    return 0     

def predict(C,S):
    W = g_wg[C]
    R = rating(C,S)
    M_rate = k*(loc_weight(C,S)*R) + ((1-k)*(g_wg[C]*R))
    #print(W," ",R," ",M_rate)
    return M_rate

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

    cos_sim = dot(cmp_set1,cmp_set2)/(norm(cmp_set1)*norm(cmp_set2))
    return cos_sim  


df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix4.1.csv")
#print(df)
#new_df = df.drop(df.columns[[0]], axis=1)

g_wg = {}
df1 = np.array(df)
glob_weight()
print("Global Weight Matrix created")
k =0.25

#Customers with NaN values along with services
M_ratings = np.argwhere(np.isnan(np.array(df)))
#print(M_ratings)

res = 0
sum1 = 0
for rate in M_ratings:
    sim_mat = {}
    for cus in df1:
        x = similarity(rate[0],int(cus[0]))
        sim_mat.update({cus[0] : x })
    sim_cus =  dict(sorted(sim_mat.items(), key=lambda item: item[1]))
    count = 0
    print("Prediction ---> ",rate[0])
    for i in sim_cus:
            res = predict(i,rate[1])
            if np.isnan(res):
                continue   
            sum1 = sum1 + res
            count+=1     
            #print('Rate of ',i,' is :',sum1," and count is : ",count)
            if count >= 10:
                break
    sum1 = int(sum1/10)
    #sum1 = round(sum1)
    df1[rate[0]][rate[1]] = int(sum1)
    print(rate[0]," ",rate[1]," ",sum1)
    print("-----------------------------------")


print(pd.DataFrame(df1))
pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_data4.1.csv")
