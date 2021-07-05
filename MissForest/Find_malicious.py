import numpy as np
import pandas as pd
import csv
import random

df1 = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Muvi\\mvi_lens_missing.csv")

def median(service):
    median_values = df1.median()  #get median values for all services
    x = median_values[service]   #store median value for <service> in x
    return x

df = df1.transpose()

mal = []

for i in range(1,318):
	#print(i)
	count = int(median(i))
	#print(count)
	for j in range(1,68):
		if df.iat[i, j] not in range(count-1,count+2):
			mal.append(df.iat[i,j])
			#print([range(count-1,count+3)])
			print(df.iat[i, j])
			#print("Position : [",i,", ",j,"]")		

print(len(mal))
print(df.size)
per_cent = ( len(mal)/df.size )  * 100

print("Malicious : ",per_cent,"%")

