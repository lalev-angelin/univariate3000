#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelin Lalev
"""
import pandas as pd
import os
import sys
from LightGBMParametersGridValidator import LightGBMParametersGridValidator
from LightGBMForecast import LightGBMForecast

### TUNING AND CONFIG
datafile_path = os.path.join("data","filtered_short.csv")
results_subdir = "series"
model_done_ext = ".done"
model_lock_ext = ".lock"
forecast_filename = "forecast.csv"
model_parameters_filename = "parameters"
logfile_name = "lightgbm_quarterly.log"
run_id = '1'


# What type of data
# We differentiate between the three types - MONTHLY, QUARTERLY AND YEARLY
# mainly because we can run them simultaneously in parallel. But we 
# provision for the possiblity that we apply deseasoning and detrending 
# and possibly more to monthly and quarterly data.
type_of_data = "QUARTERLY"

# Method
method_spec="lightgbm"

# Method parameters
params = {
    'min_data_in_leaf':list(range(2, 4)),
    'min_data_in_bin':list(range(2, 4)), 
    'max_depth':[2,4,6]
    }

### SUBROUTINES

def compute_lookback_lambda(type_of_data : str):
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
    
    return compute_lookback_periods


### START

### LOAD DATA 
data = pd.read_csv(datafile_path)

data_subset  = data[data['Type'].isin([type_of_data])]

logfile = open(logfile_name, "w")

for index,row in data_subset.iterrows():

    # Take row of data
    series_name = "%s_%s"% (row["Competition"], row["Series_Name"])
    series_category = row["Category"]
    series_type = row["Type"]
    seasonality = row["Seasonality"]
    number_of_observations = row["Number_Of_Observations"]
    number_of_predictions = row["Number_Of_Predictions"]
    total_datapoints = row["Total_Datapoints"]
    number_of_subperiods = row["Number_Of_Subperiods"]
    starting_subperiod = row["Starting_Subperiod"]
    if number_of_subperiods<1:
        number_of_subperiods = None
        starting_subperiod = None
    if starting_subperiod == None or starting_subperiod<1: 
        starting_subperiod = None
    vals=row[11:].dropna().to_list()
    
    assert len(vals)==total_datapoints, "Invalid series data, series: %s"%series_name
        
    ### Processing
    logfile.write("Now processing %s \n"%series_name)
    sys.stderr.write("Now processing %s \n"%series_name)
  
    # Compute several paths, that we need 
    model_spec = method_spec + str(run_id)
    dir_path = os.path.join(results_subdir, series_name, model_spec)
            
    lock_file_path = dir_path + model_lock_ext

    done_file_path = dir_path + model_done_ext

    forecast_file_path = os.path.join(dir_path, "forecast")

    model_param_file_path = os.path.join(dir_path, "parameters")

    # Eventually create dir and establish lock
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(done_file_path): 
        logfile.write("%s is already done. Skipping. \n"%series_name)
        sys.stderr.write("%s is already done. Skipping. \n"%series_name)
        continue

    if os.path.exists(forecast_file_path) and os.path.exists(model_param_file_path):
        sys.stderr.write("%s already has forecast. Skipping. \n"%series_name)
        logfile.write("%s already has forecast. Skipping. \n"%series_name)
        continue

    if os.path.exists(lock_file_path):
        sys.stderr.write("%s is already done. Skipping. \n"%series_name)
        logfile.write("Lock exists for %s. Skipping. \n"%series_name)
        continue
    else:
        try:
            file = open(lock_file_path, "x")
            file.close()
        except FileExistsError:
            logfile.write("Lock exists for %s. Skipping. \n", lock_file_path, series_name)
            continue 
        
    lookback_periods = compute_lookback_lambda(type_of_data)(number_of_predictions, total_datapoints)
    logfile.write("We need to forecast %d periods. \n"%number_of_predictions)
    logfile.write("And we will look back to %d periods \n"%lookback_periods)

    search = LightGBMParametersGridValidator(
        params,
        vals, 
        None, 
        forecast_horizon=number_of_predictions, 
        lookback = lookback_periods, 
        number_of_subperiods = number_of_subperiods,
        starting_subperiod = starting_subperiod, 
        apply_deseasonalizer = True,
        logfile = logfile)
   
    try:
        forecast, mape, best_params = search.search()
    except Exception:
        logfile.write("Problem with series %s"%series_name)
        logfile.write(str(Exception))
        sys.stderr.write("Problem with series %s"%series_name)
        sys.stderr.write(str(Exception))
        continue

    model = LightGBMForecast(
        vals, 
        dates=None, 
        forecast_horizon=number_of_predictions, 
        lookback = lookback_periods, 
        number_of_subperiods = number_of_subperiods,
        starting_subperiod = starting_subperiod, 
        apply_deseasonalizer = True,
        **best_params)
    
    model.generate_forecast()

    params_file = open(model_param_file_path, "w")
    for key, value in best_params.items(): 
        params_file.write("%s:%s\n"%(key, value))
    params_file.write("Validation MAPE: %f\n"%mape)
    params_file.write("Test MAPE %f\n"%model.asses_forecast_mape())
    params_file.write("Test RMSE %f\n"%model.asses_forecast_rmse())    
    params_file.close()


    output_data = pd.DataFrame(columns=["Actual", "Forecast"])
    output_data["Actual"]=vals
    output_data["Forecast"]=model.as_list()
    output_data.to_csv(forecast_file_path)

    donefile = open(done_file_path, "w")
    donefile.close()
    os.remove(lock_file_path)

logfile.close()
