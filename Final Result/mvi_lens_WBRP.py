import pandas as pd
import numpy as np
import csv
import random
from Muvi.Master1 import rating,loc_weight,glob_weight,predict,user_similarity
from statistics import mean
import math


"""
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

"""

df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")

s= df.drop(df.columns[[0]],axis=1)
g_wg = {}                                           #create an empty dictionary to store global weights  
actual = np.array(s)                                  #convert pandas dataframe to np array
s1 = np.array(s)
M_ratings = [[2,10],[2,35],[2,36],[10,5],[10,16],[10,20],[18,1],[18,2],[18,3],[18,8],[7,7],[7,9],[7,11],[8,0],[8,1],[8,2],[8,3],[8,5],[8,7],[8,8]]

l = len(M_ratings)
print(l)


for i in M_ratings:
    s1[i[0]][i[1]] = None


#s1.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_missing.csv")


print("Creating Global Matrix!!!!")
glob_weight(s1)                                       #calulating global weights
print("Global Weight Matrix created")               
k = 0

for rate in M_ratings:                              #iterate over nan locations
    res = 0
    sum_1 = 0
    sum_2 = 0
    wg = 0
    sim_mat = {}                                    #empty dictionary for similarity matrix 
    for cus in range(0,68):                                 #get the similarity for all user in M_ratings
        x = user_similarity(s1,rate[0],cus)         #with every other user in the dataset   
        sim_mat.update({cus : x })               #update the similarity scores in dictionary sim_mat
    sim_cus =  dict(sorted(sim_mat.items(), key=lambda item: item[1] , reverse = True))      #sort the dictionary in descending order
    del sim_cus[rate[0]]
    count = 0
    print("Prediction ---> ",rate[0])
    for i in sim_cus:                           #iterate over the sorted similar users upto count(count = 10)
            res , wg = predict(s1,i,rate[1],k,sim_mat)
            #print("Local prediction --->",res_loc)                              #predicting rate using user and service
            if np.isnan(res):                   #if rate is nan then ignore rest
                #print("****ignore values****")
                continue   
            sum_1 = sum_1 + res
            sum_2 = sum_2 + wg
            count+=1
            #print(i)     
            #print("count is : ",count,"Sum of local-->",sum_loc,"Sum of global--->",sum_glo)
            if count >= 15:
                break
    tot_sum = sum_1/sum_2
    
    s1[rate[0]][rate[1]] = abs(tot_sum)
    print(rate[0]," ",rate[1]," ",tot_sum)
    print("-----------------------------------")
    

print(pd.DataFrame(s1))
pd.DataFrame(s1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Final Result\\Predicted_on_observed1.csv")

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
print('MAE Value using our median model with k =',k,' is : ',res)
