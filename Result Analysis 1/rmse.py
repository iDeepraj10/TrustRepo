import csv
import numpy as np
import pandas as pd
from statistics import mean
import math

df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_missing.csv")
predicted = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\Predicted Data_kmeans.csv")
actual =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Result Analysis 1\\Muvi\\mvi_lens_imputed.csv")
predicted = predicted.drop(predicted.columns[[0]], axis=1)
#predicted = predicted.drop(predicted.columns[[0]], axis=1)
actual = actual.drop(actual.columns[[0]], axis =1)


#predicted = predicted.fillna(round(actual.mean(),2))

print(actual)
print(predicted)
actual = np.array(actual)
predicted = np.array(predicted)



M_ratings = np.argwhere(np.isnan(np.array(df)))
print(M_ratings)


rmse = []
for rate in M_ratings:
	x=np.square(np.subtract(actual[rate[0]-1][rate[1]-1],predicted[rate[0]-1][rate[1]-1]))
	print("Location : ",rate[0],"  ",rate[1]-1)
	print(x)
	print(actual[rate[0]-1][rate[1]-1],"  ",predicted[rate[0]-1][rate[1]-1])
	print("-----------------------")
	rmse.append(x)

res = mean(rmse)

rm_se = math.sqrt(res)

print('RMSE Value : ',rm_se)