#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:40:30 2024

@author: ownjo
"""

import sys
import itertools
from LightGBMForecast import LightGBMForecast
import numpy as np 

class LightGBMParametersGridValidator: 
    
    def __init__(self, 
                 timeseries: list,
                 dates: list, 
                 forecast_horizon: int = 1, 
                 lookback : int = 1, 
                 number_of_subperiods : int = None,
                 starting_subperiod : int = 1, 
                 num_leaves_range: range = None,
                 max_depth_range: range = None, 
                 learning_rate_range: range = None, 
                 n_estimators_range: range = None, 
                 subsample_for_bin_range: range = None, 
                 min_data_in_leaf: range = None, 
                 min_data_in_bin: range = None,
                 logfile = None):
                 
        self._timeseries = timeseries
        self._dates = dates
        self._forecast_horizon = forecast_horizon 
        self._lookback = lookback 
        self._number_of_subperiods = number_of_subperiods
        self._starting_subperiod = starting_subperiod 
        self._num_leaves_range = num_leaves_range
        self._max_depth_range = max_depth_range 
        self._learning_rate_range = learning_rate_range
        self._n_estimators_range = n_estimators_range
        self._subsample_for_bin_range = subsample_for_bin_range 
        self._min_data_in_leaf = min_data_in_leaf
        self._min_data_in_bin = min_data_in_bin
        self._logfile = logfile
        
 
    def validate_combination(self, 
                            **kwargs)->tuple:
        
        print(self._forecast_horizon, "\n")
        print(len(self._timeseries), "\n")
       
        assert (self._forecast_horizon * 2) < len(self._timeseries), "Timeseries too short for validation"
        
        sum_mape = 0
        
        for i in range(len(self._timeseries)-self._lookback, len(self._timeseries)):
                        
            forecast = LightGBMForecast(
                self._timeseries[:i], 
                forecast_horizon = self._forecast_horizon, 
                lookback = self._lookback, 
                number_of_subperiods = self._number_of_subperiods, 
                starting_subperiod = self._starting_subperiod, 
                **kwargs)
            
            forecast.generate_forecast()
            
            sum_mape = sum_mape + forecast.asses_forecast_mape()
        
        
        return forecast, sum_mape/self._lookback
            
    
    def search(self): 
        
        if self._logfile is not None: 
            self._logfile.write("[Info] Starting search for hyper-parameters\n")
         
        params = [self._num_leaves_range,         # 0
            self._max_depth_range,                # 1
            self._learning_rate_range,            # 2
            self._n_estimators_range,             # 3
            self._subsample_for_bin_range,        # 4
            self._min_data_in_leaf,               # 5
            self._min_data_in_bin]                # 6
        
        params = [list(lst) if lst is not None else [np.NaN] for lst in params]
        
        combinations = list(itertools.product(*params))
        
        #print(combinations)
        
        best_forecast = None
        best_combination = None
        best_mape = np.NaN

        for combination in combinations: 
            args={}
           
            if not np.isnan(combination[0]): 
                args['num_leaves']=combination[0]
            if not np.isnan(combination[1]):
                args['max_depth']=combination[1]
            if not np.isnan(combination[2]):
                args['learning_rate']=combination[2]
            if not np.isnan(combination[3]):
                args['n_estimators']=combination[3]
            if not np.isnan(combination[4]):
                args['subsample_for_bin_range']=combination[4]
            if not np.isnan(combination[5]):
                args['min_data_in_leaf']=combination[5]
            if not np.isnan(combination[6]):
                args['min_data_in_bin']=combination[6]
            
            if self._logfile is not None: 
                self._logfile.write("[INFO] Now testing\n")
                for key, value in args.items(): 
                    self._logfile.write("[INFO] %s:%s\n"%(key, value))
            
            forecast, mape = self.validate_combination(**args)
            
            self._logfile.write("[INFO] MAPE: %f\n"%mape)

            if np.isnan(best_mape) or best_mape>mape:
                best_mape = mape
                best_forecast = forecast
                best_combination = args

            if self._logfile is not None: 
                self._logfile.write("[INFO] Best MAPE %f\n"%best_mape)
                for key, value in args.items(): 
                    self._logfile.write("[INFO] %s:%s\n"%(key, value))
 
        
        return best_forecast, best_mape, best_combination


