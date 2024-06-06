#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelin Lalev
"""

import sys
import itertools
import numpy as np 
from abc import ABC, abstractmethod

class GenericParametersGridValidator(ABC): 
    
    def __init__(self, 
                 paramgrid: dict,
                 timeseries: list,
                 dates: list, 
                 forecast_horizon: int = 1, 
                 lookback : int = 1, 
                 number_of_subperiods : int = None,
                 starting_subperiod : int = 1, 
                 apply_deseasonalizer: bool = False,
                 logfile = None):
        
        self._paramgrid = paramgrid
        self._timeseries = timeseries
        self._dates = dates
        self._forecast_horizon = forecast_horizon 
        self._lookback = lookback 
        self._number_of_subperiods = number_of_subperiods
        self._starting_subperiod = starting_subperiod 
        self._apply_deseasonalizer = apply_deseasonalizer
        self._logfile = logfile
        

    @abstractmethod
    def instantiate_forecast(self, timeseries, **kwargs):
        pass


    def validate_combination(self, 
                            **kwargs)->tuple:
       
        assert (self._forecast_horizon * 2) < len(self._timeseries), "Timeseries too short for validation"
       
        sum_mape = 0
        
        for i in range(len(self._timeseries)-self._lookback, len(self._timeseries)):
                        
            forecast = self.instantiate_forecast(self._timeseries[:i], **kwargs)

            forecast.generate_forecast()
            
            sum_mape = sum_mape + forecast.asses_forecast_mape()
        
        
        return forecast, sum_mape/self._lookback
            
    
    def search(self): 
        
        if self._logfile is not None: 
            self._logfile.write("[Info] Starting search for hyper-parameters\n")
        
        best_mape = np.NaN

        combinations=[]
        for item in self._paramgrid.items():
            combinations.append([q for q in itertools.product([item[0]], item[1])])

        for combination in itertools.product(*combinations):
            
            params={key:value for key, value in combination}

            if self._logfile is not None: 
                self._logfile.write("[INFO] Now testing\n")
                for key, value in params.items(): 
                    self._logfile.write("[INFO] %s:%s\n"%(key, value))
 
            forecast, mape = self.validate_combination(**params)
            
            self._logfile.write("[INFO] MAPE: %f\n"%mape)

            if np.isnan(best_mape) or best_mape>mape:
                best_mape = mape
                best_forecast = forecast
                best_params = params


        if self._logfile is not None: 
            self._logfile.write("[INFO] Best MAPE %f\n"%best_mape)
            for key, value in best_params.items(): 
                self._logfile.write("[INFO] %s:%s\n"%(key, value))
 
        
        return best_forecast, best_mape, best_params


