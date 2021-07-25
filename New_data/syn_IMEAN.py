import pandas as pd
import numpy as np
import csv
import random
from statistics import mean
import math
 

def IMEAN(actual,missing):

    df = missing
   
    s1 = np.array(df)  
    actual = actual.drop(actual.columns[[0]], axis = 1)
    actual = np.array(actual)

    M_ratings =  np.argwhere(np.isnan(np.array(df)))     #Locations of NaN values

    df = df.drop(df.columns[[0]], axis =1)
    

 
    l = len(M_ratings)
    # print(l)


    #s1 = pd.DataFrame(s1).fillna(round(pd.DataFrame(s1).mean(axis=1),2))
    m = pd.DataFrame(s1).mean(axis=1)
    for i, col in enumerate(pd.DataFrame(s1)):
        pd.DataFrame(s1).iloc[:, i] = pd.DataFrame(s1).iloc[:, i].fillna(m)


    # print(pd.DataFrame(s1))
    s1 = pd.DataFrame(s1).drop(pd.DataFrame(s1).columns[[0]], axis =1)
    #s1 = s1.drop(s1.columns[[0]], axis =1)
    s1.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\predicted_on_2.1.1.csv")

    s1 = np.array(s1)

    # print("Prediction done")
    mae = []
    for rate in M_ratings:
        x=abs(np.subtract(s1[rate[0]-1][rate[1]-1],actual[rate[0]-1][rate[1]-1]))
        #x=np.square(np.subtract(actual[rate[0]][rate[1]],s1[rate[0]][rate[1]]))
        # print("Location : ",rate[0]-1,"  ",rate[1]-1)
        # print(x)
        # print(actual[rate[0]-1][rate[1]-1],"  ",s1[rate[0]-1][rate[1]-1])
        # print("-----------------------")
        mae.append(x)

    res = mean(mae)

    #rm_se = math.sqrt(res)
    print("MAE using imean : ",res)
    #print('RMSE Value using mean : ',rm_se)

    return res