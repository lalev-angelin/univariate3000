#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:27:35 2024

@author: ownjo
"""

from LightGBMParametersGridValidator import LightGBMParametersGridValidator

validator = LightGBMParametersGridValidator(
    [x for x in range(1, 50)], 
    dates=None,
    forecast_horizon=4,  
    lookback=6, 
    number_of_subperiods=12,
    starting_subperiod=3)

forecast, mape = validator.validate_combination(
    num_leaves=2,
    max_depth=6, 
    min_data_in_leaf=2,
    min_data_in_bin=2)