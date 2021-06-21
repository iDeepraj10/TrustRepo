import numpy as np
import pandas as pd
import csv
from missingpy import MissForest


df_org = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix2.csv")

df_null = pd.read_csv("C:\\Users\\dexter\\Desktop\\Trust and Reputation\\New folder\\Dataset\\matrix3.1.csv")

print(df_null.isnull().sum())
M_ratings = np.argwhere(np.isnan(np.array(df)))

# Make an instance and perform the imputation
imputer = MissForest()

for rate in M_ratings:
	X = df_org.drop(rate[1], axis=1)
	X_imputed = imputer.fit_transform(X)