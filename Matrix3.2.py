import csv
import pandas as pd
import numpy as np

with open('matrix3.2.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix2.csv")


df1 = df.mask(np.random.random(df.shape) < .2)
print(df1)

df1.to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix3.2.csv")  
