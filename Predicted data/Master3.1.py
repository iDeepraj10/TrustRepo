###############################################################
#####RATING PREDICTION IN THE CONTEXT OF SERVICE DISCOVERY#####
###############################################################
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial import distance
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from numpy import dot
from numpy.linalg import norm
import csv




def rating(customer,service):
    return df1[int(customer)][int(service)]


def predict_user(C,S):
    R = rating(C,S)
    W = sim_user.get(C)
    M_rate = W * R 
    #print(W," <- weight",R," <- rate",M_rate)
    return M_rate

def predict_item(C,S):
    R = rating(C,S)
    W = sim_item.get(C)
    M_rate = W * R
    return M_rate

df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Dataset\\matrix 3.1.1.csv")
#print(df)

df1 = np.array(df)

#Customers with NaN values along with services
M_ratings = np.argwhere(np.isnan(np.array(df)))
print(M_ratings)

#get similarity for two customers
def user_similarity(c1,c2):
    temp_set = []
    count = -1
    for i,j in zip(df1[c1],df1[c2]): 
        count+=1  
        if np.isnan(i) or np.isnan(j):
            continue
        temp_set.append(count)

    cmp_set1 = []
    cmp_set2 = []
    for i in temp_set:
        cmp_set1.append(df1[c1][i])    
        cmp_set2.append(df1[c2][i])

    cos_sim = np.corrcoef(cmp_set1, cmp_set2)
    return cos_sim[0,1]  

#similarity of two items
def item_similarity(i1,i2):
    temp_set = []
    count = -1
    df2 = df1.transpose()

    for i,j in zip(df2[i1],df2[i2]):
        count+=1  
        if np.isnan(i) or np.isnan(j):
            continue
        temp_set.append(count)
    cmp_set1 = []
    cmp_set2 = []
    for i in temp_set:
        cmp_set1.append(df2[i1][i])    
        cmp_set2.append(df2[i2][i])    
    cos_sim = np.corrcoef(cmp_set1, cmp_set2)
    return cos_sim[0,1]


res1 = 0
res2 = 0
sum1 = 0
sum2 = 0
sim_user = {}
sim_item = {}
c = M_ratings.shape[0]
for rate in M_ratings:
    k = 0.9
    for cus in df1:
        x = user_similarity(rate[0],int(cus[0]))
        sim_user.update({cus[0] : x })
        y = item_similarity(rate[1],int(cus[0]))
        sim_item.update({cus[0] : y })
        #print("service 1 : ",rate[0]," service 2 : ",int(cus[0])," most similar : ",x)
    sim_cus =  dict(sorted(sim_user.items(), key=lambda item: item[1]),reverse=True)
    sim_item =  dict(sorted(sim_item.items(), key=lambda item: item[1], reverse = True))
    count = 0
    for i,j in zip(sim_cus,sim_item):
            res1 = predict_user(i,rate[1])
            res2 = predict_item(rate[0],j)
            if np.isnan(res1) or np.isnan(res2) or i==0 or j==0:
                continue   
            sum1 = sum1 + res1
            sum2 = sum2 + res2
            count+=1     
            if count >= 10:
                break
    sum1 = sum1/10
    sum2 = sum2/10
    pred = round(abs(( k * sum1)) + abs(((1-k)*sum2)),2)
    df1[rate[0]][rate[1]] = pred
    #print("Loading...     Wait for : ",c," more secs")
    c = c - 1
    print(rate[0]," ",rate[1]," = ",pred)
    print("----------------------------------")


print(pd.DataFrame(df1))
pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_data3.5.csv")
