#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:40:30 2024

@author: ownjo
"""

from LightGBMForecast import LightGBMForecast
import sys
from sklearn.model_selection import TimeSeriesSplit

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
                 **kwargs):
        
                 
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
        self._kwargs = kwargs
       
 
    def validate_combination(self, 
                            **kwargs)->float:
        
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
        None
        
        