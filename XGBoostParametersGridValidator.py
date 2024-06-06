#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelin Lalev
"""

import sys
import itertools
from GenericParametersGridValidator import GenericParametersGridValidator 
from XGBoostForecast import XGBoostForecast
import numpy as np 

class XGBoostParametersGridValidator(GenericParametersGridValidator): 
    
    def instantiate_forecast(self, timeseries, **kwargs):
        return  XGBoostForecast(
                    timeseries, 
                    forecast_horizon = self._forecast_horizon,
                    lookback = self._lookback,
                    number_of_subperiods = self._number_of_subperiods,
                    starting_subperiod = self._starting_subperiod,
                    **kwargs)

