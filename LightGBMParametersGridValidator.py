#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelin Lalev
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
                 apply_deseasonalizer: bool = False,
                 num_leaves_ran: list = None,
                 max_depth_ran: list = None, 
                 learning_rate_ran: list = None, 
                 n_estimators_ran: list = None, 
                 subsample_for_bin_ran: list = None, 
                 min_data_in_leaf_ran: list = None, 
                 min_data_in_bin_ran: list = None,
                 logfile = None):
                 
        self._timeseries = timeseries
        self._dates = dates
        self._forecast_horizon = forecast_horizon 
        self._lookback = lookback 
        self._number_of_subperiods = number_of_subperiods
        self._starting_subperiod = starting_subperiod 
        self._apply_deseasonalizer = apply_deseasonalizer
        self._num_leaves_ran = num_leaves_ran
        self._max_depth_ran = max_depth_ran 
        self._learning_rate_ran = learning_rate_ran
        self._n_estimators_ran = n_estimators_ran
        self._subsample_for_bin_ran = subsample_for_bin_ran 
        self._min_data_in_leaf_ran = min_data_in_leaf_ran
        self._min_data_in_bin_ran = min_data_in_bin_ran
        self._logfile = logfile
        
 
    def validate_combination(self, 
                            **kwargs)->tuple:
       
        assert (self._forecast_horizon * 2) < len(self._timeseries), "Timeseries too short for validation"
        
        sum_mape = 0
        
        for i in range(len(self._timeseries)-self._lookback, len(self._timeseries)):
                        
            forecast = LightGBMForecast(
                self._timeseries[:i], 
                forecast_horizon = self._forecast_horizon, 
                lookback = self._lookback, 
                number_of_subperiods = self._number_of_subperiods, 
                starting_subperiod = self._starting_subperiod, 
                apply_deseasonalizer=self._apply_deseasonalizer,
                **kwargs)
            
            forecast.generate_forecast()
            
            sum_mape = sum_mape + forecast.asses_forecast_mape()
        
        
        return forecast, sum_mape/self._lookback
            
    
    def search(self): 
        
        if self._logfile is not None: 
            self._logfile.write("[Info] Starting search for hyper-parameters\n")
         
        params = [self._num_leaves_ran,         # 0
            self._max_depth_ran,                # 1
            self._learning_rate_ran,            # 2
            self._n_estimators_ran,             # 3
            self._subsample_for_bin_ran,        # 4
            self._min_data_in_leaf_ran,         # 5
            self._min_data_in_bin_ran]          # 6
        
        params = [lst if lst is not None else [np.NaN] for lst in params]
        
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


