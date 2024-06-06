#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelin Lalev
"""

import sys
import itertools
from GenericParametersGridValidator import GenericParametersGridValidator 
from LightGBMForecast import LightGBMForecast
import numpy as np 

class LightGBMParametersGridValidator(GenericParametersGridValidator): 
    
    def instantiate_forecast(self, **kwargs):
        return  LightGBMForecast(self._timeseries, **kwargs)

