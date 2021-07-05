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
import centroid

g_wg = {} 
df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Dataset\\matrix 3.3.3.csv")

#function returns medium value of a service
center_values = centroid.cen(df.drop(df.columns[[0]], axis =1))  #get median values for all services

#print(median_values)



def rating(df1,customer,service):
    #print(customer," ",service)
    #print(customer,"    ",service)
    return df1[int(customer)][int(service)]

    

#function returns weight of a customer for specific service
def loc_weight(df,c,s):
    Central_point = center_values[s-1]      #get median of <s>
    #print(Central_point)
    Rating = rating(df,c,s)            #return 
    Weight = 1 - abs((Central_point - Rating))/10
    #print(Weight," ",Central_point," ",Rating," of :",c," ",s)
    return Weight

def glob_weight(df1):
    sum1 = 0
    count = 0
    for c1 in range(0,100):                         #iterate from 0 to 99
        for cus in range(0,100):                             #
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

def predict_global(df,C,S):
    R = rating(df,C,S)
    M_rate = g_wg[C]*R
    #print(W," ",R," ",M_rate)
    return M_rate

def predict_local(df,C,S):
    R = rating(df,C,S)
    M_rate = loc_weight(df,C,S)*R 
    #print(W," ",R," ",M_rate)
    return M_rate

#get similarity for two customers
def similarity(df1,c1,c2):
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
"""
#Load the dataset
df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix C.csv")
#print(df)
 
                                          #create an empty dictionary to store global weights  
df1 = np.array(df)                                  #convert pandas dataframe to np array
glob_weight()                                       #calulating global weights
print("Global Weight Matrix created")               
k =0.25                



M_ratings = np.argwhere(np.isnan(np.array(df)))     #Locations of NaN values
#print(M_ratings)


for rate in M_ratings:                              #iterate over nan locations
    res_loc = 0
    res_glo = 0
    sum_loc = 0
    sum_glo = 0
    tot_sum = 0
    sim_mat = {}                                    #empty dictionary for similarity matrix 
    for cus in df1:                                 #get the similarity for all user in M_ratings
        x = similarity(rate[0],int(cus[0]))         #with every other user in the dataset   
        sim_mat.update({cus[0] : x })               #update the similarity scores in dictionary sim_mat
    sim_cus =  dict(sorted(sim_mat.items(), key=lambda item: item[1] , reverse = True))      #sort the dictionary in descending order
    del sim_cus[rate[0]]
    count = 0
    print("Prediction ---> ",rate[0])
    for i in sim_cus:                           #iterate over the sorted similar users upto count(count = 10)
            res_loc = predict_local(i,rate[1])
            print("Local prediction --->",res_loc)                              #predicting rate using user and service
            res_glo = predict_global(i,rate[1])
            print("Global prediction--->",res_glo)
            if np.isnan(res_loc) or np.isnan(res_glo):                   #if rate is nan then ignore rest
                print("****ignore values****")
                continue   
            sum_loc = sum_loc + res_loc
            sum_glo = sum_glo + res_glo
            count+=1
            print(i)     
            print("count is : ",count,"Sum of local-->",sum_loc,"Sum of global--->",sum_glo)
            if count >= 10:
                break
    tot_sum = ((k*sum_loc/10) + (1-k)*sum_glo/10)
    tot_sum = round(tot_sum,2)
    
    df1[rate[0]][rate[1]] = tot_sum
    print(rate[0]," ",rate[1]," ",tot_sum)
    print("-----------------------------------")


print(pd.DataFrame(df1))
pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_data4.1.csv")


"""
