import pandas as pd
from syn_IMEAN import IMEAN
from syn_UMEAN import UMEAN
from syn_IPCC import IPCC
from syn_UPCC import UPCC
from syn_WSRec import WSRec


actual = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 2.1.csv")
missing = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\New_data\\matrix 2.1.3.csv") 

print("Predicting IPCC")
mae_on_IPCC = IPCC(actual,missing)
print("IPCC prediction done")
print("Predicting UPCC")
mae_on_UPCC = UPCC(actual,missing)
print("UPCC prediction done")
print("Predicting WSRec")
mae_on_WSRec = WSRec(actual,missing)
print("WSRec prediction done")
print("Predicting IMEAN")
mae_on_IMEAN = IMEAN(actual,missing)
print("IMEAN prediction done")
print("Predicting UMEAN")
mae_on_UMEAN = UMEAN(actual,missing)
print("UMEAN prediction done")


print("Result Table :")

print("mae_on_IPCC      : ",mae_on_IPCC)
print("mae_on_UPCC      : ",mae_on_UPCC)
print("mae_on_WSRec     : ",mae_on_WSRec)
print("mae_on_IMean     : ",mae_on_IMean)
print("mae_on_UMEAN     : ",mae_on_UMEAN)