#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelin Lalev
"""

import sys
import itertools
from GenericParametersGridValidator import GenericParametersGridValidator 
from CatBoostForecast import CatBoostForecast 
import numpy as np 

class CatBoostParametersGridValidator(GenericParametersGridValidator): 
    
    def instantiate_forecast(self, **kwargs):
        return  CatBoostForecast(self._timeseries, **kwargs)

