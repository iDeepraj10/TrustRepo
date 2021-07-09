import csv
import numpy as np
import pandas as pd
from statistics import mean
import math


predicted = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_IPCC.csv")
actual =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")


predicted = predicted.drop(predicted.columns[[0]], axis=1)
predicted = predicted.drop(predicted.columns[[0]], axis=1)
actual = actual.drop(actual.columns[[0]], axis =1)


#predicted = predicted.fillna(round(actual.mean(),2))

print(actual)
print(predicted)
actual = np.array(actual)
predicted = np.array(predicted)



M_ratings = [[2,10],[2,35],[2,36],[10,5],[10,16],[10,20],[18,1],[18,2],[18,3],[18,8],[7,7],[7,9],[7,11]]
print(M_ratings)


mae = []
for rate in M_ratings:
	x = abs(np.subtract(predicted[rate[0]][rate[1]-1],actual[rate[0]][rate[1]]))
	print("Location : ",rate[0],"  ",rate[1])
	print(x)
	print(actual[rate[0]][rate[1]],"  ",predicted[rate[0]][rate[1]-1])
	print("-----------------------")
	mae.append(x)

res = mean(mae)



print('MAE Value : ',res)