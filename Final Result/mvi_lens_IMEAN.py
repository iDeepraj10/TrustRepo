import pandas as pd
import numpy as np
import csv
import random
from statistics import mean
import math
 

df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")
s1 = df.drop(df.columns[[0]], axis =1)


actual = np.array(s1)               

s1 = np.array(s1)

M_ratings = [[2,10],[2,35],[2,36],[10,5],[10,16],[10,20],[18,1],[18,2],[18,3],[18,8],[7,7],[7,9],[7,11],[8,0],[8,1],[8,2],[8,3],[8,5],[8,7],[8,8]]

l = len(M_ratings)
print(l)


for i in M_ratings:
    s1[i[0]][i[1]] = None


s1 = pd.DataFrame(s1).fillna(round(pd.DataFrame(s1).mean(axis=1),2))
# m = pd.DataFrame(s1).mean(axis=1)
# for i, col in enumerate(pd.DataFrame(s1)):
#     pd.DataFrame(s1).iloc[:, i] = pd.DataFrame(s1).iloc[:, i].fillna(m)


print(pd.DataFrame(s1))
#s1 = pd.DataFrame(s1).drop(pd.DataFrame(s1).columns[[0]], axis =1)
pd.DataFrame(s1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\movie_lens_imean.csv")

s1 = np.array(s1)

print("Prediction done")
mae = []
for rate in M_ratings:
    x=abs(np.subtract(s1[rate[0]][rate[1]],actual[rate[0]][rate[1]]))
    #x=np.square(np.subtract(actual[rate[0]][rate[1]],s1[rate[0]][rate[1]]))
    print("Location : ",rate[0],"  ",rate[1])
    print(x)
    print(actual[rate[0]][rate[1]],"  ",s1[rate[0]][rate[1]])
    print("-----------------------")
    mae.append(x)

res = mean(mae)

#rm_se = math.sqrt(res)
print("MAE using imean : ",res)
#print('RMSE Value using mean : ',rm_se)