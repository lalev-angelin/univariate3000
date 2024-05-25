#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ownjo
"""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller

### TUNING AND CONFIG
data_loc = os.path.join("data","timeseries.csv")
output_loc = os.path.join("series", "adfstatistic.csv")

### LOAD DATA
data = pd.read_csv(data_loc)

### PROCESS DATA

results = pd.DataFrame(columns=["Series_Name", "C_ADF", "CT_ADF", "CTT_ADF", "N_ADF", "C_PVALUE", "CT_PVALUE", "CTT_PVALUE", "N_PVALUE", "C_CV1", "CT_CV1", "CTT_CV1", "N_CV1", "C_CV5", "CT_CV5", "CTT_CV5", "N_CV5"])
#print(results)
#sys.exit(1)

for index,row in data.iterrows():
    # process a row of the frame 
    series_name = "%s_%s"% (row["Competition"], row["Series_Name"])
    series_category = row["Category"]
    series_type = row["Type"]
    seasonality = row["Seasonality"]
    number_of_observations = row["Number_Of_Observations"]
    number_of_predictions = row["Number_Of_Predictions"]
    total_datapoints = row["Total_Datapoints"]
    data=row[9:].dropna().to_list()

    result = adfuller(data)
    # Print the results
    #print('ADF Statistic:', result[0])
    #print('p-value:', result[1])
    #print('Critical Values:')
    #for key, value in result[4].items():
    #    print(f'   {key}: {value}')

    result_c=[]
    result_c.append(result[0]) # ADF Statistic
    result_c.append(result[1]) # p-value
    result_c.append(result[4].get("1%"))  # Crit. value 1%
    result_c.append(result[4].get("5%"))  # Crit. value 5%
    
    result = adfuller(data, regression="ct")
    result_ct=[]
    result_ct.append(result[0]) # ADF Statistic
    result_ct.append(result[1]) # p-value
    result_ct.append(result[4].get("1%"))  # Crit. value 1%
    result_ct.append(result[4].get("5%"))  # Crit. value 5%

    result = adfuller(data, regression="ctt")
    result_ctt=[]
    result_ctt.append(result[0]) # ADF Statistic
    result_ctt.append(result[1]) # p-value
    result_ctt.append(result[4].get("1%"))  # Crit. value 1%
    result_ctt.append(result[4].get("5%"))  # Crit. value 5%

    result = adfuller(data, regression="n")
    result_n=[]
    result_n.append(result[0]) # ADF Statistic
    result_n.append(result[1]) # p-value
    result_n.append(result[4].get("1%"))  # Crit. value 1%
    result_n.append(result[4].get("5%"))  # Crit. value 5%
 
    new_row= pd.DataFrame({
        "Series_Name": [series_name], 
        "C_ADF": [result_c[0]], 
        "CT_ADF": [result_ct[0]], 
        "CTT_ADF": [result_ctt[0]], 
        "N_ADF": [result_n[0]], 
        "C_PVALUE": [result_c[1]],
        "CT_PVALUE": [result_ct[1]], 
        "CTT_PVALUE": [result_ctt[1]], 
        "N_PVALUE": [result_n[1]], 
        "C_CV1": [result_c[2]],
        "CT_CV1": [result_ct[2]],
        "CTT_CV1": [result_ctt[2]],
        "N_CV1": [result_n[2]],
        "C_CV5": [result_c[3]], 
        "CT_CV5": [result_ct[3]], 
        "CTT_CV5": [result_ctt[3]],
        "N_CV5": [result_n[3]]}) 
    results = pd.concat([results, new_row], ignore_index=True)
    #print(results)
    #sys.exit(1)
    print(index, file=sys.stderr)

results.to_csv(output_loc)
