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
    
    def instantiate_forecast(self, **kwargs):
        return  XGBoostForecast(self._timeseries, **kwargs)

