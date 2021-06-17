import csv
import random
import pandas as pd

with open('matrix2.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    count=1
 
    arr=[]
    for i in range(1000):
        col = [] 
        rate=1
        for j in range(1000):
            if(i==0):
                col.append(rate)
                rate=rate+1
                continue
            col.append(abs(random.choice([ele for ele in range(rate-3,rate+3) if ele !=0 if ele <=10])))
            if(count%100==0):
                rate=rate+1
            count=count+1    
        arr.append(col)

    print(pd.DataFrame(arr))    
    writer.writerows(arr)

        
