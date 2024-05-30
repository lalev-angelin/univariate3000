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

### PROCESS DIRS
for d in dirs:
    series_name = d
    d = os.path.join(series_dir, d) 
    if os.path.isdir(d):
        modeldirs = os.listdir(d)
        for modeldir in modeldirs:
            modeldir = os.path.join(d, modeldir)
            if os.path.isdir(modeldir):
                filepaths = os.listdir(modeldir)
                for filepath in filepaths:
                    if filepath=="model":
                        sys.stderr.write("Processing %s ...\n"%modeldir)
                        
                        modelpath=os.path.join(modeldir, "model")
                        datapath=os.path.join(modeldir, "forecast")

                        ### LOAD THE DATA FILE 
                        forecast = pd.read_csv(datapath)
                        
                        ### LOAD THE MODEL IN PICKLE FILE
                        regressor = pickle.load(open(modelpath, "rb"))
                         
                        splits = series_name.split("_")
                        competition = splits[0]
                        series = splits[1]
                        query = "Series_Name == '%s' and Competition == '%s'"%(series,competition) 
                        #print(query) 
                        row = data.query(query)
                        #print(row)
                        #print(row.iloc[0]["Number_Of_Predictions"])

                        if ("Type == 'YEARLY'"): 
                            input_length = row.iloc[0]["Number_Of_Predictions"]
                            output_length = row.iloc[0]["Number_Of_Predictions"] 
                        
                        if ("Type = 'QUARTERLY'"):
                            input_length = 4 
                            output_length = row.iloc[0]["Number_Of_Predictions"]


                        if ("Type == 'MONTHLY'"):
                            continue 
                            input_length = 12
                            output_length = row.iloc[0]["Number_Of_Predictions"]

                        if ("Type == 'DAILY'"):
                            continue 
                            input_length = 30
                            output_length = row.iloc[0]["Number_Of_Predictions"]

                        if ("Type == 'WEEKLY'"): 
                            continue
                            input_length = 53
                            output_length = row.iloc[0]["Number_Of_Predictions"]

                        if ("Type == 'HOURLY'"):
                            continue
                            input_length = 24*30
                            output_length = row.iloc[0]["Number_Of_Predictions"]

                        if ("Type == 'OTHER'"):
                            continue 
                            input_length =  len(vals)//4
                            output_length = row.iloc[0]["Number_Of_Predictions"]


                        input_segments, output_segments = computeSlidingWindows(forecast['Actual'],
                                input_length, output_length)

                        #print(input_segments)
                        #print(output_segments)

                        trainX  = np.array(input_segments[:-number_of_predictions])
                        testX = np.array(input_segments[-1:])
                        trainY = np.array(output_segments[:-number_of_predictions])
                        testY = np.array(output_segments[-1:])

                        multi_output_regressor = MultiOutputRegressor(regressor)
                        multi_output_regressor.fit(trainX, trainY)

                        ### PREDICT
                        predictY = multi_output_regressor.predict(input_segments)

                        forecast = []
                        for output_segment in predictY:
                            forecast.append(predictY[0])

                        for forecast_element_index in range(1, len(predictY[-1])-1):
                            forecast.append(predictY[-1][forecast_element_index])

                        ### SAVE
                        output_data = pd.DataFrame(columns=["Actual", "Projection"])
                        output_data["Actual"]=data["Actual"]
                        output_data["Forecast"]=forecast
                        output_data.to_csv("fix_forecast")

                        sys.exit(1)  
