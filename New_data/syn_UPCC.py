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

def UPCC(actual,missing):

    def rating(df1,customer,service):
        return df1[int(customer)][int(service)]


    def predict_user(df1,C,S):
        R = rating(df1,C,S)
        W = sim_user.get(C)
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
    for rate in M_ratings:
        res1 =0
        sum1 = 0
        sum2 = 0
        S=0
        for cus in df1:
            x = user_similarity(df1,rate[0],int(cus[0]))
            sim_user.update({cus[0] : x })
        sim_cus =  dict(sorted(sim_user.items(), key=lambda item: item[1],reverse=True))
        count = 0
        del sim_cus[rate[0]]
        for i in sim_cus :
                #print("calling predict_user with ",i," and ",rate[1])
                res1,S = predict_user(df1,i,rate[1])

                if np.isnan(res1) or i==0 :
                    #print("***********Result Ignored*************")
                    continue       
                sum1 = sum1 + abs(res1)
                sum2 = sum2 + abs(S)
                count+=1 
                #print(res1," ",sum1," ",count)
                #print(S," ",sum2," ",count)    
                #print("-------result considered-----------")
                if count >= 20:
                    break
        f_sum = sum1/abs(sum2)
        #print(sum1,"+",sum2,"=",f_sum)
        pred = round(abs(f_sum),2)
        df1[rate[0]][rate[1]] = pred
        print(rate[0]," ",rate[1]," = ",pred)
        print("----------------------------------")


    # print(pd.DataFrame(df1))
    pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\Predicted data\\Predicted_UPCC.csv")

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
    print("MAE using UPCC : ",res)
    #print('RMSE Value using mean : ',rm_se)

    return res