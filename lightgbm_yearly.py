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
from lightgbm import LGBMRegressor
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

# LGBMRegressor parameters
gbmparams = {"num_leaves":31,
             "max_depth":-1,
             "learning_rate":0.1,
             "n_estimators":1000}

### LOAD DATA 
data = pd.read_csv(datafile_path)

yearly_data = data[data['Type'].isin(["YEARLY"])]

for index,row in yearly_data.iterrows():
    
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
    model_spec ="ligthgbm_model_numleaves_%d_maxdepth_%d_learningrate_%d_nestimators_%d"%(gbmparams["num_leaves"], gbmparams["max_depth"], gbmparams["learning_rate"], gbmparams["n_estimators"])

    dir_path = os.path.join(results_subdir, series_name, model_spec)
    dir_path = dir_path.replace("-1", "NIL") 
    #print(dir_path)
            
    model_file_path = os.path.join(dir_path, "model")

    lock_file_path = model_file_path + model_lock_ext
    #print(lock_file_path)

    done_file_path = model_file_path + model_done_ext
    #print(done_file_path)

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
            number_of_predictions,
            number_of_predictions)
    

    #print(len(input_segments))
    #print(input_segments)
    #print(len(output_segments))
    #print(output_segments)
    
    trainX  = np.array(input_segments[:-1])
    testX = np.array(input_segments[-1:])
    trainY = np.array(output_segments[:-1])
    testY = np.array(output_segments[-1:])

    #print(trainX.shape)
    #print(testX.shape)
    #print(trainY.shape)
    #print(testY.shape)
    #print(testY)
    #print(trainY)
    #sys.exit(1)

    ### FIT
    regressor = LGBMRegressor(boosting_type="gbdt",
                                  num_leaves=gbmparams['num_leaves'],
                                  max_depth=gbmparams['max_depth'],
                                  learning_rate=gbmparams['learning_rate'],
                                  n_estimators=gbmparams['n_estimators'])
 
    multi_output_regressor = MultiOutputRegressor(regressor)

    multi_output_regressor.fit(trainX, trainY)

    ### SAVE MODEL
    file = open(model_file_path, "wb")
    pickle.dump(regressor, file)
    file.close()

    ### PREDICT
    predictY = pd.DataFrame(multi_output_regressor.predict(testX))
    print(predictY)

    ### SAVE
    output_data = pd.DataFrame(columns=["Actual", "Projection"])
    output_data["Actual"]=vals
    vals=vals[:-len(predictY)]
    print(vals)
    vals.extend(predictY)
    print(vals)
    output_data["Projection"]=vals

    print(output_data)
    sys.exit(1)


    sys.exit(1) 

    predictions.append(predictY[predictY.columns[0]].iloc[0])

    # We got the last prediction first, so reverse
    predictions.reverse()
    print(predictions)

    predictY = pd.DataFrame(regressor.predict())
    predictY = pd.concat([predictY[:-12], pd.Series(predictions)]).reset_index()
    print(predictY)

