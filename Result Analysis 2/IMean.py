import csv
import numpy as np
import pandas as pd
from statistics import mean
import math



actual =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix B.csv")
df =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix C.csv")


actual = actual.drop(actual.columns[[0]], axis =1)
M_ratings = np.argwhere(np.isnan(np.array(df)))
print(M_ratings)
predicted = actual

m = predicted.mean(axis=1)
for i, col in enumerate(predicted):
	predicted.iloc[:, i] = predicted.iloc[:, i].fillna(m)

predicted.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix IMean.csv")


print(actual)
print(predicted)
actual = np.array(actual)
predicted = np.array(predicted)



rmse = []
for rate in M_ratings:
	x=np.square(np.subtract(actual[rate[0]-1][rate[1]-1],predicted[rate[0]-1][rate[1]-1]))
	print("Location : ",rate[0],"  ",rate[1])
	print(x)
	print(actual[rate[0]-1][rate[1]-1],"  ",predicted[rate[0]-1][rate[1]-1])
	print("-----------------------")
	rmse.append(x)
print(rmse)

res = mean(rmse)

rm_se = math.sqrt(res)

print('RMSE Value : ',rm_se)
