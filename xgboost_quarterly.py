#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ownjo
"""
import pandas as pd
import os
import sys
import pickle
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from subroutines import computeSlidingWindows 

### TUNING AND CONFIG
datafile_path = os.path.join("data","timeseries.csv")
results_subdir = "series"
model_done_ext = ".done"
model_lock_ext = ".lock"
forecast_filename = "forecast.csv"

# LGBMRegressor default parameters. Included for clarity only...
gbmparams = {
             "max_depth":6,
             "learning_rate":0.1,
             "n_estimators":1000}

# What type of data
type_of_data = "QUARTERLY"

# Model spec template 
model_spec_template ="xgboost_model_maxdepth_%d_learningrate_%d_nestimators_%d"

# Set the regressor type
regressor_type = "xgboost"

### START

# Look as many periods back as you are required to predict forward
if type_of_data == "YEARLY":
    # As many years back as you need to forecast forward
    compute_lookback_periods = lambda forward_look, overall_length : forward_look
elif type_of_data == "QUARTERLY":
    # Four periods back
    compute_lookback_periods = lambda forward_look, overall_length : 4
elif type_of_data == "MONTHLY":
    # Twelve periods back
    compute_lookback_periods = lambda forward_look, overall_length : 12
elif type_of_data == "WEEKLY":
    # 53 periods back
    compute_lookback_periods = lambda forward_look, overall_length : 53
elif type_of_data == "DAILY":
    # 31 days back
    compute_lookback_periods = lambda forward_look, overall_length : 30
elif type_of_data == "HOURLY":
    # 168 hours back (a week)
    compute_lookback_periods = lambda forward_look, overall_length : 168
elif type_of_data == "OTHER":
    # 1/8 of the total periods
    compute_lookback_periods = lambda forward_look, overall_length : overall_length//8
else:
    sys.stderr.write("Invalid type_of_data value")
    sys.exit(1)


### LOAD DATA 
data = pd.read_csv(datafile_path)

yearly_data = data[data['Type'].isin([type_of_data])]

for index,row in yearly_data.iterrows():

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
        model_spec = model_spec_template%(gbmparams["max_depth"], gbmparams["learning_rate"], gbmparams["n_estimators"])

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
       

        ### COMPUTE FORWARD (HOW FAR WE FORECAST) AND BACKWARD (HOW BACK WE LOOK)
        ### "HORIZON" 
        forward_horizon = number_of_predictions
        backward_horizon = compute_lookback_periods(number_of_predictions, len(vals))

        ### PREPARE TRAIN AND TEST SET
        input_segments, output_segments = computeSlidingWindows(vals, 
            backward_horizon,
            forward_horizon)
    
        trainX = input_segments[:-forward_horizon]
        trainY = output_segments[:-forward_horizon]
        #print("TrainX\n", trainX)
        #print("TrainY\n", trainY)

        # The regressor does not support vector output.
        # So we need to do it element by element 
        regressors = []
        for period in range(0, forward_horizon): 
            trainYElement = [segment[period] for segment in trainY]

            ### FIT
            if regressor_type == 'lightgbm': 
                    regressor = LGBMRegressor(boosting_type="gbdt",
                                          num_leaves=gbmparams['num_leaves'],
                                          max_depth=gbmparams['max_depth'],
                                          learning_rate=gbmparams['learning_rate'],
                                          n_estimators=gbmparams['n_estimators']
                                          )
            elif regressor_type == 'xgboost':
                    regressor = XGBRegressor(
                                          max_depth=gbmparams['max_depth'],
                                          learning_rate=gbmparams['learning_rate'],
                                          n_estimators=gbmparams['n_estimators']
                                          )
            elif regressor_type == 'catboost':
                    regressor = CatBoostRegressor(
                                          max_depth=gbmparams['max_depth'],
                                          learning_rate=gbmparams['learning_rate'],
                                          n_estimators=gbmparams['n_estimators']
                                          )
            else: 
                sys.stderr.write("Invalid choice of regressor")
                sys.exit(20)

            regressor.fit(trainX, trainYElement)

            ### SAVE MODEL
            file = open(model_file_path + str(period+1), "wb")
            pickle.dump(regressor, file)
            file.close()

            regressors.append(regressor)

        ### PREDICT
        regressor = regressors[0]
        predictY = regressor.predict(input_segments).tolist()
        for i in range(1, forward_horizon):
            regressor = regressors[i]
            predictY.append(regressor.predict(input_segments[-1:])[0])

        predictY = ['N/A'] * backward_horizon + predictY 
        #print(predictY)

        ### SAVE
        output_data = pd.DataFrame(columns=["Actual", "Forecast"])
        output_data["Actual"]=vals
        output_data["Forecast"]=predictY
        output_data.to_csv(forecast_file_path)

        file=open(done_file_path, "x")
        file.close()

        os.remove(lock_file_path)
    except Exception as e:
        sys.stderr.write("Error processing %s \n"%series_name)
        sys.stderr.write(str(e))

