import pandas as pd
import numpy as np
import csv
import random

df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\ratings.csv")

df1= df.drop('timestamp',axis='columns')

s=df1.pivot(*df1.columns)
  

s = s[s.columns[s.isnull().mean() < 0.9]]
s = s.loc[s.isnull().mean(axis=1).lt(0.6)]
#print(s.isnull().mean(axis=1))


M_ratings = np.argwhere(np.isnan(np.array(s)))
mis = len(M_ratings)
tot = s.size
print("No. of missing values : ",mis)
print("Total size of data : ",s.size)
print(s.info())

res = (mis/tot) * 100
print(res)
s.columns =[ele for ele in range(0,317)]
s.index =[ele for ele in range(0,68)]

#s.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")