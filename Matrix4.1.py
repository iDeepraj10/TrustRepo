import pandas as pd
import numpy as np
import csv
import random


df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 1.csv")
df = df.drop(df.columns[[0]], axis =1)

count = 1
ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
for row, col in random.sample(ix, int(round(.1*len(ix)))):
    if col in range(0,21):
        count = 1
    elif col in range(21,41):
        count = 2
    elif col in range(41,61):
        count = 3
    elif col in range(61,81):
        count = 4
    elif col in range(81,101):
        count = 5
    elif col in range(101,121):
        count = 6
    elif col in range(121,141):
        count = 7
    elif col in range(141,161):
        count = 8
    elif col in range(161,181):
        count = 9
    elif col in range(181,201):
        count = 10                                    
    df.iat[row, col] = random.choice([ele for ele in range(count-2,count+3) if ele <11 if ele >0])

df.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 2.1.csv")

df1 = df.mask(np.random.random(df.shape) < .1)
print(df1)

df1.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 2.1.1.csv")  
