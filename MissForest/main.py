import numpy as np
import pandas as pd
import csv
import random

col = 10

arr_r = random.choice([ele for ele in range(1,11) if ele not in range(col-2,col+3)])
print([ele for ele in range(1,11) if ele not in range(col-2,col+3)])

print(arr_r)