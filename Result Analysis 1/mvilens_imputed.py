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

#s.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")  

s = s[s.columns[s.isnull().mean() < 0.9]]
s = s.loc[s.isnull().mean(axis=1).lt(0.6)]
#print(s.isnull().mean(axis=1))


M_ratings = np.argwhere(np.isnan(np.array(s)))


#print("No. of missing values : ",mis)
#print("Total size of data : ",s.size)
print(s.info())
mis = len(M_ratings)
tot = s.size

res = (mis/tot) * 100
print(res)


s.columns =[ele for ele in range(0,317)]
s.index =[ele for ele in range(0,68)]

g_wg = {}                                           #create an empty dictionary to store global weights  
s1 = np.array(s)                                    #convert pandas dataframe to np array

observed = []
empty=[]

for j in range(0,317):
    for i in range(0,68):
        if not np.isnan(s1[i][j]):
            observed.append(s1[i][j])    
    s[j].fillna(random.choice([ele for ele in observed]), inplace=True)        

print(s)
pd.DataFrame(s).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_imputed.csv")

df1 = pd.DataFrame(s1).mask(np.random.random(s1.shape) < .1)

pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_missing10.csv")
