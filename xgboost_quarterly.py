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
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.multioutput import MultiOutputRegressor

from subroutines import computeSlidingWindows 

### TUNING AND CONFIG
datafile_path = os.path.join("data","timeseries.csv")
results_subdir = "series"
model_done_ext = ".done"
model_lock_ext = ".lock"
forecast_filename = "forecast.csv"

# LGBMRegressor parameters
gbmparams = {"num_leaves":31,
             "max_depth":6,
             "learning_rate":0.1,
             "n_estimators":1000}

### LOAD DATA 
data = pd.read_csv(datafile_path)

quarterly_data = data[data['Type'].isin(["QUARTERLY"])]

for index,row in quarterly_data.iterrows():
   
    try:

        # Take row of data
        series_name = "%s_%s"% (row["Competition"], row["Series_Name"])
        series_category = row["Category"]
        series_type = row["Type"]
        seasonality = row["Seasonality"]
        number_of_observations = row["Number_Of_Observations"]
        number_of_predictions = row["Number_Of_Predictions"]
        total_datapoints = row["Total_Datapoints"]
        vals=row[8:].dropna().to_list()
        #print(number_of_predictions)
        #print(vals)

        ### Processing
        sys.stderr.write("Now processing %s \n"%series_name)

        # Compute several paths, that we need 
        model_spec ="xgboost_model_maxdepth_%d_learningrate_%d_nestimators_%d"%(gbmparams["max_depth"], gbmparams["learning_rate"], gbmparams["n_estimators"])

        dir_path = os.path.join(results_subdir, series_name, model_spec)
        dir_path = dir_path.replace("-1", "NIL") 
            
        model_file_path = os.path.join(dir_path, "model")

        lock_file_path = model_file_path + model_lock_ext

        done_file_path = model_file_path + model_done_ext

        forecast_file_path = os.path.join(dir_path, "forecast")

        # Eventually create dir and establish lock
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if os.path.exists(done_file_path): 
            sys.stderr.write("%s is already done. Skipping. \n"%series_name)
            continue

        if os.path.exists(lock_file_path):
            sys.stderr.write("Lock exists for %s. Skipping. \n"%series_name)
            continue
        else:
            try:
                file = open(lock_file_path, "x")
                file.close()
            except FileExistsError:
                sys.stderr.write("Lock exists for %s. Skipping. \n", model_lock_path, series_name)
                continue 
        
        ### PREPARE TRAIN AND TEST SET

        input_segments, output_segments = computeSlidingWindows(vals, 
                4,
                number_of_predictions)
    
        trainX  = np.array(input_segments[:-number_of_predictions])
        testX = np.array(input_segments[-1:])
        trainY = np.array(output_segments[:-number_of_predictions])
        testY = np.array(output_segments[-1:])

        ### FIT
        regressor = XGBRegressor(
                                  learning_rate=gbmparams['learning_rate'],
                                  max_depth=gbmparams['max_depth'],
                                  n_estimators=gbmparams['n_estimators'],
                                  gpu_id=0)
 
        multi_output_regressor = MultiOutputRegressor(regressor)

        multi_output_regressor.fit(trainX, trainY)

        ### SAVE MODEL
        file = open(model_file_path, "wb")
        pickle.dump(regressor, file)
        file.close()

        ### PREDICT
        predictY = multi_output_regressor.predict(testX)

        ### SAVE
        output_data = pd.DataFrame(columns=["Actual", "Projection"])
        output_data["Actual"]=vals
    
        vals=vals[:-len(predictY[0])]
    
        vals.extend(predictY[0])
    
        output_data["Projection"]=vals
        output_data.to_csv(forecast_file_path)

        file=open(done_file_path, "x")
        file.close()
    
        os.remove(lock_file_path)
    except:
        sys.stderr.write("Error processing %s \n"%series_name)
