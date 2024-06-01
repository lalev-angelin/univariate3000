#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ownjo
"""
import pandas as pd
import os
import sys
import numpy as np
import pickle
from sklearn.multioutput import MultiOutputRegressor
from subroutines import computeSlidingWindows 
from subroutines import computeMAPE
from subroutines import computeRMSE
import string

### TUNING AND CONFIG
series_dir = os.path.join("series")
data_file = os.path.join("data", "timeseries.csv")

### ENUMERATE FILES
dirs = os.listdir(series_dir)

### LOAD THE ORIGINAL SOURCE DATA

# We need it because it has the forecast horizon
data = pd.read_csv(data_file)
data['CatBoost_MAPE']=np.NaN
data['CatBoost_RMSE']=np.NaN
data['LightGBM_MAPE']=np.NaN
data['LightGBM_RMSE']=np.NaN
data['XGBoost_MAPE']=np.NaN
data['XGBoost_RMSE']=np.NaN

for index, rw in data.iterrows():
    #print(rw)
    filename = os.path.join(series_dir, "%s_%s"%(rw['Competition'], rw['Series_Name']))
    filename
    if os.path.isdir(filename):
        modeldirs = os.listdir(filename)
        #print(modeldirs)
        for modeldir in modeldirs:
            if os.path.isdir(os.path.join(filename, modeldir)):
                #print(modeldir)
                splits = modeldir.split("_")
                #print(splits)
                #exit(1)
                if (splits[0]=="lightgbm"):
                    model_name = splits[0]
                    max_depth = splits[5]
                    learning_rate = splits[7]
                    num_leaves = splits[3]
                    number_of_estimators = splits[9]
                    #print("model_name", model_name, "\n")
                    #print("num_leaves", num_leaves, "\n")
                    #print("max_depth", max_depth, "\n")
                    #print("learning_rate", learning_rate, "\n")
                    #print("number_of_estimators", number_of_estimators, "\n")
                elif (splits[0]=="xgboost"):
                    #print(splits)
                    model_name = splits[0]
                    max_depth = splits[3]
                    num_leaves = "N/A"
                    learning_rate = splits[5]
                    number_of_estimators = splits[7]
                    #print("model_name", model_name, "\n")
                    #print("num_leaves", num_leaves, "\n")
                    #print("max_depth", max_depth, "\n")
                    #print("learning_rate", learning_rate, "\n")
                    #print("number_of_estimators", number_of_estimators, "\n")
                elif (splits[0]=="catboost"):
                    #print(splits)
                    model_name = splits[0]
                    max_depth = splits[3]
                    num_leaves = "N/A"
                    learning_rate = splits[5]
                    number_of_estimators = splits[7]
                    #print("model_name", model_name, "\n")
                    #print("num_leaves", num_leaves, "\n")
                    #print("max_depth", max_depth, "\n")
                    #print("learning_rate", learning_rate, "\n")
                    #print("number_of_estimators", number_of_estimators, "\n")
                else:
                    continue
           

            num_forecasts = data.iloc[index]['Number_Of_Predictions']
            forecast = pd.read_csv(os.path.join(filename, modeldir, "forecast")) 

            if (model_name=="catboost"): 
                data['CatBoost_MAPE'][index]=computeMAPE(forecast['Actual'], forecast['Forecast'], num_forecasts)
                data['CatBoost_RMSE'][index]=computeRMSE(forecast['Actual'], forecast['Forecast'], num_forecasts)
            elif(model_name=="lightgbm"):
                data['LightGBM_MAPE'][index]=computeMAPE(forecast['Actual'], forecast['Forecast'], num_forecasts)
                data['LightGBM_RMSE'][index]=computeRMSE(forecast['Actual'], forecast['Forecast'], num_forecasts)
            elif(model_name=="xgboost"):
                data['XGBoost_MAPE'][index]=computeMAPE(forecast['Actual'], forecast['Forecast'], num_forecasts)
                data['XGBoost_RMSE'][index]=computeRMSE(forecast['Actual'], forecast['Forecast'], num_forecasts)
            
    else: 
        stderr.write("No model dir for series %s"%rw['Series_Name'])
        sys.exit(1)

data.to_csv('data_with_metrics.csv') 
data1 = data[['Competition', 'Series_Name', 'CatBoost_MAPE', 'CatBoost_RMSE', 'LightGBM_MAPE', 'LightGBM_RMSE', 'XGBoost_MAPE', 'XGBoost_RMSE']]
data1.to_csv('data_with_metrics_shortened.csv')
