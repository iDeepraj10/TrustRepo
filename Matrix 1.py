import csv
import pandas as pd

count=1
 
arr=[]
for i in range(100):
    col = []
    rate=1
    for l in range(200):
        # if(i==0):
        #     col.append(rate)
        #     rate=rate+1
        #     continue
        col.append(rate)
        if(count%20==0):
            rate=rate+1
        count=count+1    
            
    arr.append(col)
print(arr)

pd.DataFrame(arr).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 1.csv")  

        
