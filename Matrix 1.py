import csv

with open('matrix A.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    count=1
 
    arr=[]
    for i in range(101):
        col = []
        rate=1
        for l in range(100):
                col.append(rate)
                if(i==0):
                    rate=rate+1
                    continue
                if(count%10==0):
                    rate=rate+1
                count=count+1    
            
        arr.append(col)
    print(arr)

    writer.writerows(arr)

        
