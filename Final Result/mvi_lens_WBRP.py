import pandas as pd
import numpy as np
import csv
import random
from Muvi.Master import rating,loc_weight,glob_weight,predict_global,predict_local,similarity
from statistics import mean
import math


df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\ratings.csv")

df1= df.drop('timestamp',axis='columns')

s=df1.pivot(*df1.columns)
  

s = s[s.columns[s.isnull().mean() < 0.9]]
s = s.loc[s.isnull().mean(axis=1).lt(0.6)]
#print(s.isnull().mean(axis=1))
 

M_ratings = []
mis = len(M_ratings)
tot = s.size
#print("No. of missing values : ",mis)
#print("Total size of data : ",s.size)
#print(s.info())

#res = (mis/tot) * 100

s.columns =[ele for ele in range(0,317)]
s.index =[ele for ele in range(0,68)]
s.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")

g_wg = {}                                           #create an empty dictionary to store global weights  
actual = np.array(s)                                  #convert pandas dataframe to np array
s1 = np.array(s)
M_ratings = [[2,10],[2,35],[2,36],[10,5],[10,16],[10,20],[18,1],[18,2],[18,3],[18,8],[7,7],[7,9],[7,11]]

l = len(M_ratings)
print(l)


for i in M_ratings:
    s1[i[0]][i[1]] = None


#s1.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_missing.csv")


print("Creating Global Matrix!!!!")
glob_weight(s1)                                       #calulating global weights
print("Global Weight Matrix created")               
k =0.10                

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
    #del sim_cus[rate[0]]
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
            if count >= 15:
                break
    tot_sum = ((k*sum_loc/15) + (1-k)*sum_glo/15)
    tot_sum = round(tot_sum,2)
    
    s1[rate[0]][rate[1]] = abs(tot_sum)
    print(rate[0]," ",rate[1]," ",tot_sum)
    print("-----------------------------------")
    

print(pd.DataFrame(s1))
pd.DataFrame(s1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\Predicted_on_observed.csv")

print("Prediction done")
rmse = []
for rate in M_ratings:
    x=abs(np.subtract(s1[rate[0]][rate[1]],actual[rate[0]][rate[1]]))
    #x=np.square(np.subtract(actual[rate[0]][rate[1]],s1[rate[0]][rate[1]]))
    print("Location : ",rate[0],"  ",rate[1])
    print(x)
    print(actual[rate[0]][rate[1]],"  ",s1[rate[0]][rate[1]])
    print("-----------------------")
    rmse.append(x)

res = mean(rmse)

#rm_se = math.sqrt(res)
print('MAE Value using our model : ',res)