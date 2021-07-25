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
np.seterr(divide='ignore', invalid='ignore')

#######################FUNCTIONS#############################

def WSRec(actual,missing) :

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


    df = missing
    #print(df)
    df1 = np.array(df)
    actual = actual.drop(actual.columns[[0]],axis = 1)
    actual = np.array(actual)
   
    #Customers with NaN values along with services
    M_ratings = np.argwhere(np.isnan(np.array(df)))     #Locations of NaN values

    # df = df.drop(df.columns[[0]],axis = 1)
    


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
        for cus1,cus2 in zip (range(0,100),range(0,200)):
            x = user_similarity(df1,rate[0],cus1)
            sim_user.update({cus1 : x })
            y = item_similarity(df1,rate[1],cus2)
            sim_item.update({cus2 : y })
        sim_cus =  dict(sorted(sim_user.items(), key=lambda item: item[1] , reverse = True))
        sim_item = dict(sorted(sim_item.items(), key=lambda item: item[1] , reverse = True))
        count = 0
        del sim_cus[rate[0]]
        del sim_item[rate[0]]
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


    # print(pd.DataFrame(df1))
    pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_WSRec.csv")

    # print("Prediction done")
    mae = []

    for rate in M_ratings:
        x=abs(np.subtract(df1[rate[0]-1][rate[1]-1],actual[rate[0]-1][rate[1]-1]))
        #x=np.square(np.subtract(actual[rate[0]][rate[1]],s1[rate[0]][rate[1]]))
        # print("Location : ",rate[0],"  ",rate[1])
        # print(x)
        # print(actual[rate[0]-1][rate[1]-1],"  ",df1[rate[0]-1][rate[1]-1])
        # print("-----------------------")
        mae.append(x)

    res = np.mean(mae)

    #rm_se = math.sqrt(res)
    print("MAE using umean : ",res)
    #print('RMSE Value using mean : ',rm_se)

    return res