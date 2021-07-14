import csv
import random
import pandas as pd

with open('C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 2.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    count=1
 
    arr=[]
    for i in range(201):
        col = [] 
        rate=1
        for j in range(100):
            if(i==0):
                col.append(rate)
                rate=rate+1
                continue
            col.append(abs(random.choice([ele for ele in range(rate-3,rate+3) if ele !=0 if ele <=10])))
            if(count%20==0):
                rate=rate+1
            count=count+1    
        arr.append(col)

    print(pd.DataFrame(arr))    
    writer.writerows(arr)

        
