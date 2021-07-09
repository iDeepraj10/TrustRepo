import csv
import numpy as np
import pandas as pd
from statistics import mean
import math



actual =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")
#df =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix C.csv")


actual = actual.drop(actual.columns[[0]], axis =1)
M_ratings = [[2,10],[2,35],[2,36],[10,5],[10,16],[20,0],[10,20],[18,1],[18,2],[18,3],[18,8],[7,7],[7,9],[7,11]]
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



mae = []
for rate in M_ratings:
	x=abs(np.subtract(actual[rate[0]][rate[1]-1],predicted[rate[0]][rate[1]-1]))
	print("Location : ",rate[0],"  ",rate[1])
	print(x)
	print(actual[rate[0]-1][rate[1]-1],"  ",predicted[rate[0]-1][rate[1]-1])
	print("-----------------------")
	mae.append(x)


res = mean(mae)

print('MAE Value by IMEAN: ',res)
