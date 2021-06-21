import csv
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import math

actual = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix2.csv")
predicted = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_data3.2.csv")


predicted = predicted.drop(predicted.columns[[0]], axis=1)
predicted = predicted.drop(predicted.columns[[0]], axis=1)

print(predicted)

mse = metrics.mean_squared_error(actual, predicted)

rmse = math.sqrt(mse)

print('RMSE Value : ',rmse)
