import csv
import random

with open('matrix2.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    count=1
 
    arr=[]
    for i in range(30):
        col = [] 
        rate=1
        for j in range(30):
      #      if(i==0):
      #          col.append(rate)
      #          rate=rate+1
      #          continue
            col.append(abs(random.choice([ele for ele in range(rate-3,rate+3) if ele !=0 if ele <=10])))
            if(count%8==0):
                rate=rate+1
            count=count+1    
        arr.append(col)

    writer.writerows(arr)

        
