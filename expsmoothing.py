import pandas as pd
import os
import sys
from statsmodels.tsa.stattools import pacf

### TUNING & CONFIG
data_loc = os.path.join("data", "timeseries.csv")

### IMPORT THE DATASET

data=pd.read_csv(data_loc, header=0)

### COMPUTE PACF WITH DIFFERENT LAGS
for index,row in data.iterrows():
    values = list(row[9:].dropna())
    cors = pacf(values,len(values)//2-1)   #:-p 
    cors = cors[2:]
    if any([cor>0.7 for cor in cors]):
        t = [cor>0.7 for cor in cors]
        print(t)
        print("================================================")
        print(values)
        print("================================================")
        print(cors) 
        sys.exit(1)
    

