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


#######################FUNCTIONS#############################

def rating(df1,customer,service):
    #print("returning rate for ",customer,' ',service)
    return df1[int(customer)][int(service)]


def predict_user(df1,C,S):
    R = rating(df1,int(C),S)
    W = sim_user.get(C)
    M_rate = W * R 
    #print("Predicted for ",C," with ",W," and ",R)
    return M_rate,W

def predict_item(df1,C,S):
    R = rating(df1,int(C),S)
    W = sim_item.get(C)
    M_rate = W * R
    #print("Predicted for ",C," with ",W," and ",R)
    return M_rate,W

#get similarity for two customers
def user_similarity(df1,c1,c2):
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
def item_similarity(df1,i1,i2):
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


###############################################################################


df = pd.read_csv( "C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\mvi_lens.csv")
df = df.drop(df.columns[[0]],axis = 1)
print(df)

df1 = np.array(df)

#Customers with NaN values along with services
M_ratings = [[2,10],[2,35],[2,36],[10,5],[10,16],[10,20],[18,1],[18,2],[18,3],[18,8],[7,7],[7,9],[7,11]]
print(M_ratings)




sim_user = {}
sim_item = {}

for rate in M_ratings:
    k = 0.90
    res1 = 0
    res2 = 0
    sum1 = 0
    sum2 = 0
    Sim1 = 0
    Sim2 = 0
    for cus in range(0,68):
        x = user_similarity(df1,rate[0],cus)
        sim_user.update({cus : x })
        y = item_similarity(df1,rate[1],cus)
        sim_item.update({cus : y })
    sim_cus =  dict(sorted(sim_user.items(), key=lambda item: item[1] , reverse = True))
    sim_item = dict(sorted(sim_item.items(), key=lambda item: item[1] , reverse = True))
    count = 0
    del sim_cus[rate[0]]
    del sim_item[rate[1]]
    for i,j in zip(sim_cus,sim_item):
            #print("calling predict_user with ",i," and ",rate[1])
            res1,S1 = predict_user(df1,i,rate[1])
            #print("calling predict_item with ",j," and ",rate[1])
            res2,S2 = predict_item(df1,j,rate[1])
            
            if np.isnan(res1) or np.isnan(res2) or i==0 or j==0:
                #print("***********Result Ignored*************")
                continue       
            sum1 = sum1 + abs(res1)
            Sim1 = Sim1 + abs(S1)
            sum2 = sum2 + abs(res2)
            Sim2 = Sim2 + abs(S2)
            count+=1     
            #print("-------result considered-----------")
            if count >= 35 :
                break
    f_sum1 = sum1/Sim1
    f_sum2 = sum2/Sim2
    pred = round(( k * f_sum1) + ((1-k)*f_sum2),2)
    df1[rate[0]][rate[1]] = pred
    print(rate[0]," ",rate[1]," = ",pred)
    print("----------------------------------")


print(pd.DataFrame(df1))
pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_WSRec.csv")
