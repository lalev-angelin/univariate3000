#!/usr/bin/env python3
# _*_ coding: utf-8 -*-
"""
@author: Angelin Lalev
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math 

######################################################################
# Most of our models get vectors as input and output. This requires 
# input and output data of our timeseries to be organized in such 
# overlapping segments/vectors.

def computeSlidingWindows(datalist: list[float], input_segment_length: int, output_segment_length: int) -> np.ndarray: 

    # We start by spliting the whole row into "training" windows of 
    # input_segment_length. Keep in mind that if you pick  list of say 7 
    # elements  and 3 input, you will have 5 such segments.
    # [1 2 3 4 5 6 7] -> [1 2 3], [2 3 4], [3 4 5], [4 5 6], [5 6 7]
    # This is because you can get as high as 5 and have the two last
    # numbers to compose your last segment.
    # So the higest position from which segment may be made is
    # the length of the row minus the input_length, decreased by one.
    input_segments = []

    for start in range(0, len(datalist)-(input_segment_length-1)):
        input_segments.append(datalist[start:start+input_segment_length])

    # Exactly the same procedure for the output segments

    output_segments = []

    for start in range(0, len(datalist)-(output_segment_length-1)):
        output_segments.append(datalist[start:start+output_segment_length])

    # We remove ___output_length__ items from the end of
    # input segments to allign them with the output segments.
    # We remove __input_length_segments___ items from __output_segments__
    input_segments = input_segments[:-output_segment_length]
    output_segments = output_segments[input_segment_length:]
 
    return(np.array(input_segments), np.array(output_segments))

def computeRMSE(actual, forecast, position): 
    return math.sqrt(mean_squared_error(actual[:-position], forecast[:-position]))

def computeMAPE(actual, forecast, position): 
    return mean_absolute_percentage_error(actual[:-position], forecast[:-position ]) 
