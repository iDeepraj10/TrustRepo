import pandas as pd
import numpy as np
import csv
import random

with open('matrix4.1.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

df = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix2.csv")

ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
for row, col in random.sample(ix, int(round(.1*len(ix)))):
    df.iat[row, col] = random.choice([ele for ele in range(1,10) if ele != col if ele != range(col-3,col+3) if ele <11])

df1 = df.mask(np.random.random(df.shape) < .1)

df1.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix4.1.csv")  
