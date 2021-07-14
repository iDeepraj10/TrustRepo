import csv
import numpy as np
import pandas as pd
from statistics import mean
import math

df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 2.1.1.csv")
predicted = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_data2.1.1.csv")
actual =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 2.1.csv")
predicted = predicted.drop(predicted.columns[[0]], axis=1)
predicted = predicted.drop(predicted.columns[[0]], axis=1)
actual = actual.drop(actual.columns[[0]], axis =1)


print(actual)
print(predicted)
actual = np.array(actual)
predicted = np.array(predicted)


M_ratings = np.argwhere(np.isnan(np.array(df)))
print(M_ratings)


mae = []
for rate in M_ratings:
	x=abs(np.subtract(actual[rate[0]][rate[1]-1],predicted[rate[0]][rate[1]-1]))
	print("Location : ",rate[0],"  ",rate[1])
	print(x)
	print(actual[rate[0]][rate[1]-1],"  ",predicted[rate[0]][rate[1]-1])
	print("-----------------------")
	mae.append(x)

res = mean(mae)

mae = math.sqrt(res)

print('MAE Value : ',mae)