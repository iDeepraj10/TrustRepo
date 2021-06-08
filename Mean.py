import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv( r'C:\Users\dexter\Desktop\Trust and Reputation\Dataset\matrix3.1.csv')



mean_df = df.fillna(value = df.mean(),axis = 0)

mean_error = np.sqrt(mean_squared_error(mean_df.values.flatten(),df.values.flatten()))

print(mean_error)                     

