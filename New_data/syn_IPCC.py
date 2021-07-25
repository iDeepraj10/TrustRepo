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

def IPCC(actual,missing):
    def rating(df1,customer,service):
        return df1[int(customer)][int(service)]


    def predict_item(df1,C,S):
        R = rating(df1,C,S)
        W = sim_item.get(C)
        M_rate = W * R
        print("Predicted for ",C," with ",W," and ",R)
        return M_rate,W
     

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
    # df = df.drop(df.columns[[0]],axis = 1)

    actual = actual.drop(actual.columns[[0]],axis = 1)
    actual = np.array(actual)
    
    #Customers with NaN values along with services
    M_ratings = np.argwhere(np.isnan(np.array(df)))     #Locations of NaN values

    # print(df)
    df1 = np.array(df)

    sim_item = {}

    for rate in M_ratings:
        for cus in df1:
            sum1 = 0
            sum2 = 0
            S=0
            y = item_similarity(df1,rate[1]-1,int(cus[0]))
            sim_item.update({cus[0] : y })
        sim_item = dict(sorted(sim_item.items(), key=lambda item: item[1],reverse =True))
        count = 0
        #del sim_item[rate[1]-1]
        for i in sim_item :
                #print("calling predict_item with ",i," and ",rate[1])
                res2,S = predict_item(df1,i,rate[1]-1)
                
                if np.isnan(res2) or i==0:
                    #print("***********Result Ignored*************")
                    continue       
                sum2 = sum2 + abs(res2)
                sum1 = sum1 + abs(S)
                count+=1     
                #print("-------result considered-----------")
                if count >= 20:
                    break
        f_sum = sum2/sum1
        pred = round(f_sum,2)
        df1[rate[0]][rate[1]] = pred
        print(rate[0]," ",rate[1]-1," = ",pred)
        print("----------------------------------")


    #print(pd.DataFrame(df1))
    pd.DataFrame(df1).to_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\Predicted_IPCC.csv")

    #print("Prediction done")
    mae = []

    for rate in M_ratings:
        x=abs(np.subtract(df1[rate[0]-1][rate[1]-1],actual[rate[0]-1][rate[1]-1]))
        #x=np.square(np.subtract(actual[rate[0]][rate[1]],s1[rate[0]][rate[1]]))
        #print("Location : ",rate[0],"  ",rate[1])
        #print(x)
        #print(actual[rate[0]-1][rate[1]-1],"  ",s1[rate[0]-1][rate[1]-1])
        #print("-----------------------")
        mae.append(x)

    res = np.mean(mae)

    #rm_se = math.sqrt(res)
    print("MAE using IPCC : ",res)
    #print('RMSE Value using mean : ',rm_se)


    return res