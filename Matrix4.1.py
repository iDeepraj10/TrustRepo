import pandas as pd
import numpy as np
import csv
import random


df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix A.csv")
count = 1
ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
for row, col in random.sample(ix, int(round(.1*len(ix)))):
    if col in range(1,11):
        count = 1
    elif col in range(11,21):
        count = 2
    elif col in range(21,31):
        count = 3
    elif col in range(31,41):
        count = 4
    elif col in range(41,51):
        count = 5
    elif col in range(51,61):
        count = 6
    elif col in range(61,71):
        count = 7
    elif col in range(71,81):
        count = 8
    elif col in range(81,91):
        count = 9
    elif col in range(91,101):
        count = 10                                    
    df.iat[row, col] = random.choice([ele for ele in range(1,11) if ele not in range(count-2,count+3) if ele <11 if ele >0])

df.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix B.csv")

df1 = df.mask(np.random.random(df.shape) < .4)
print(df1)

df1.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix C.csv")  
