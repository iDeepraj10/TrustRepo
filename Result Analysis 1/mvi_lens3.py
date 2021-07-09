import pandas as pd
import numpy as np
import csv
import random
from Muvi.Master import rating,loc_weight,glob_weight,predict_global,predict_local,similarity
from statistics import mean
import math

actual = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_imputed10.csv")

s1 = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_missing10.csv")


print(actual)
M_ratings = np.argwhere(np.isnan(np.array(s1)))

s1 = np.array(s1)
print("Creating Global Matrix!!!!")
glob_weight(s1)                                       #calulating global weights
print("Global Weight Matrix created")               
k =0.25                



#M_ratings = np.argwhere(np.isnan(np.array(s)))     #Locations of NaN values
#print(M_ratings)


for rate in M_ratings:                              #iterate over nan locations
    res_loc = 0
    res_glo = 0
    sum_loc = 0
    sum_glo = 0
    tot_sum = 0
    sim_mat = {}                                    #empty dictionary for similarity matrix 
    for cus in range(0,68):                                 #get the similarity for all user in M_ratings
        x = similarity(s1,rate[0],cus)         #with every other user in the dataset   
        sim_mat.update({cus : x })               #update the similarity scores in dictionary sim_mat
    sim_cus =  dict(sorted(sim_mat.items(), key=lambda item: item[1] , reverse = True))      #sort the dictionary in descending order
    del sim_cus[rate[0]]
    count = 0
    print("Prediction ---> ",rate[0])
    for i in sim_cus:                           #iterate over the sorted similar users upto count(count = 10)
            res_loc = predict_local(s1,i,rate[1])
            #print("Local prediction --->",res_loc)                              #predicting rate using user and service
            res_glo = predict_global(s1,i,rate[1])
            #print("Global prediction--->",res_glo)
            if np.isnan(res_loc) or np.isnan(res_glo):                   #if rate is nan then ignore rest
                #print("****ignore values****")
                continue   
            sum_loc = sum_loc + res_loc
            sum_glo = sum_glo + res_glo
            count+=1
            #print(i)     
            #print("count is : ",count,"Sum of local-->",sum_loc,"Sum of global--->",sum_glo)
            if count >= 10:
                break
    tot_sum = ((k*sum_loc/10) + (1-k)*sum_glo/10)
    tot_sum = round(tot_sum,2)
    
    s1[rate[0]][rate[1]] = abs(tot_sum)
    print(rate[0]," ",rate[1]," ",tot_sum)
    print("-----------------------------------")


print(pd.DataFrame(s1))
pd.DataFrame(s1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\Predicted Data10.csv")
print("Prediction done")
actual = actual.drop(actual.columns[[0]], axis =1)

actual = np.array(actual)

rmse = []
for rate in M_ratings:
    x=np.square(np.subtract(actual[rate[0]][rate[1]],s1[rate[0]][rate[1]]))
    print("Location : ",rate[0],"  ",rate[1])
    print(x)
    print(actual[rate[0]][rate[1]],"  ",s1[rate[0]][rate[1]])
    print("-----------------------")
    rmse.append(x)

res = mean(rmse)

rm_se = math.sqrt(res)

print('RMSE Value : ',rm_se)
