import csv
import numpy as np
import pandas as pd
from statistics import mean
import math

actual = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix1.csv")
predicted = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_data4.1.csv")
df =  pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix4.1.csv")

predicted = predicted.drop(predicted.columns[[0]], axis=1)
predicted = predicted.drop(predicted.columns[[0]], axis=1)

actual = np.array(actual)
predicted = np.array(predicted)


M_ratings = np.argwhere(np.isnan(np.array(df)))

rmse = []
for rate in M_ratings:
	x=np.square(np.subtract(actual[rate[0]-1][rate[1]-1],predicted[rate[0]-1][rate[1]-1]))
	x = math.sqrt(x)
	print(x)
	rmse.append(x)

print('RMSE Value : ',mean(rmse))