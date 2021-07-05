import csv
import numpy as np
import pandas as pd
from statistics import mean
import math


actual =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix B.csv")
df =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix C.csv")
df = df.drop(df.columns[[0]], axis =1)
#predicted = actual.drop(actual.columns[[0]], axis =1)
predicted = df
print(predicted)
predicted = predicted.fillna(round(df.mean(),2))
predicted = predicted.drop(predicted.columns[[0]], axis =1)
predicted.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix UMean.csv")


print(predicted)
actual = np.array(actual)
predicted = np.array(predicted)


M_ratings = np.argwhere(np.isnan(np.array(df)))


rmse = []
for rate in M_ratings:
	x=np.square(np.subtract(actual[rate[0]-1][rate[1]-1],predicted[rate[0]-1][rate[1]-1]))
	print("Location : ",rate[0],"  ",rate[1])
	print(x)
	print(actual[rate[0]-1][rate[1]-1],"  ",predicted[rate[0]-1][rate[1]-1])
	print("-----------------------")
	rmse.append(x)

res = mean(rmse)

rm_se = math.sqrt(res)

print('RMSE Value : ',rm_se)