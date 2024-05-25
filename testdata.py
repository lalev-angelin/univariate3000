#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ownjo
"""
import pandas as pd
import os
import sys

### TUNING AND CONFIG
data_loc = os.path.join("data","timeseries.csv")

### LOAD DATA 
data = pd.read_csv(data_loc)

for index,row in data.iterrows():
    # Вземаме ред с данни
    series_name = "%s_%s"% (row["Competition"], row["Series_Name"])
    series_category = row["Category"]
    series_type = row["Type"]
    seasonality = row["Seasonality"]
    number_of_observations = row["Number_Of_Observations"]
    number_of_predictions = row["Number_Of_Predictions"]
    total_datapoints = row["Total_Datapoints"]
    data=row[8:].dropna().to_list()
#    print(data)
    if len(data)!=total_datapoints:
        print(series_name, len(data), total_datapoints)
        
    if total_datapoints!=number_of_observations+number_of_predictions:
        print(series_name, number_of_observations, number_of_predictions, total_datapoints)
