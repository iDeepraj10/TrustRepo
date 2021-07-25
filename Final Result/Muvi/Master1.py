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


df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Final Result\\Muvi\\mvi_lens_missing.csv")
g_wg = {} 


#function returns medium value of a service
median_values = df.median()  #get median values for all services
#print(median_values)



def rating(df1,customer,service):
    #print(customer," ",service)
    return df1[int(customer)][int(service)]

    

#function returns weight of a customer for specific service
def loc_weight(df,c,s):
    Central_point = median_values[s]      #get median of <s>
    #print(Central_point)
    Rating = rating(df,c,s)            #return 
    Weight = 1 - abs((Central_point - Rating))/10
    #print(Weight," ",Central_point," ",Rating," of :",c," ",s)
    return Weight
 
def glob_weight(df1):
    sum1 = 0
    count = 0
    for c1 in range(0,68):                         #iterate from 0 to 99
        for cus in range(0,317):                             #
            wl = loc_weight(df1,c1,cus)       #get local weight for every user
            if math.isnan(wl):                      #if value is nan then ignore
                wl = 0
                continue
            #print("loc weight for ",cus[0]," of ",c1," is ",wl)
            sum1 = sum1 + wl
            count = count + 1
            res = sum1/count
        g_wg.update({c1 : res})                     #adding the weight to a dictionary in a key-value format
    return 0     

def predict(df,C,S,k,sim_mat):
    R = rating(df,C,S)
    g_w = g_wg[C]
    l_w = sim_mat.get(C)
    tot = ((k*l_w) + (1-k)*g_w) 
    #print(W," ",R," ",M_rate)
    return tot * R , tot

#get similarity for two customers
def user_similarity(df1,c1,c2):
    temp_set = []                       #temporary empty set
    count = -1                          #count keeps track of services not having null
    for i,j in zip(df1[c1],df1[c2]):    #iterate over values of user c1 and user c2 with i and j respectively
        count+=1                        
        if np.isnan(i) or np.isnan(j):  
            continue
        temp_set.append(count)          #store the service number if both user have rated that service

    cmp_set1 = []                       
    cmp_set2 = []                       #Two empty set to compare the ratings of c1 and c2
    for i in temp_set:
        cmp_set1.append(df1[c1][i])    
        cmp_set2.append(df1[c2][i])

    cos_sim = np.corrcoef(cmp_set1, cmp_set2)       #get the similarity between two users using pearson's coefficient
    return cos_sim[0,1]  

#similarity of two items
def item_similarity(df1,i1,i2):
    temp_set = []
    count = -1
    df2 = df1.transpose()

    for i,j in zip(df2[i1],df2[i2]):
        count+=1  
        if np.isnan(i) or np.isnan(j):
            continue
        temp_set.append(count)
    cmp_set1 = []
    cmp_set2 = []
    for i in temp_set:
        cmp_set1.append(df2[i1][i])    
        cmp_set2.append(df2[i2][i])    
    cos_sim = np.corrcoef(cmp_set1, cmp_set2)
    return cos_sim[0,1]    

def item_predict(df1, C , S,k):
    R = rating(df1,C,S)
    g_w = g_wg[C]
    l_w = loc_weight(df1,C,S)
    tot = ((k*l_w) + (1-k)*g_w)
    print("Rate of ",C," with weight ",tot,"for service ",S," is ",R)
    return tot * R , tot   