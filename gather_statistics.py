#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ChatGPT, Angelin Lalev 
"""

import os
import pandas as pd
import re
import sys

# Function to read the contents of the parameters file and extract Test MAPE
def read_parameters_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    parameters = {}
    test_mape = None
    max_depth = None
    learning_rate = None
    min_data_in_bin = None
    min_data_in_leaf = None

    for line in lines:
        if "Test MAPE" in line:
            splits = line.split(" ")
            test_mape = float(splits[2].strip())
        if "max_depth" in line: 
            splits = line.split(":")
            max_depth = int(splits[1].strip())
        if "learning_rate" in line: 
            splits = line.split(":")
            learning_rate = float(splits[1].strip())
        if "min_data_in_bin" in line: 
            splits = line.split(":")
            min_data_in_bin = int(splits[1].strip())
        if "min_data_in_leaf" in line: 
            splits = line.split(":")
            min_data_in_split = int(splits[1].strip())


    parameters_string = ''.join(lines)
    return test_mape, max_depth, learning_rate, min_data_in_bin, min_data_in_leaf, parameters_string

# Function to process the directory structure and extract the required information
def process_directories(root_dir):
    data = []

    for series_name in os.listdir(root_dir):
        series_path = os.path.join(root_dir, series_name)
        if os.path.isdir(series_path):
            for method_name in os.listdir(series_path):
                method_path = os.path.join(series_path, method_name)
                if os.path.isdir(method_path):
                    parameters_file_path = os.path.join(method_path, 'parameters')
                    if os.path.exists(parameters_file_path):
                        test_mape, max_depth, learning_rate, min_data_in_bin, min_data_in_leaf, parameters_string = read_parameters_file(parameters_file_path)
                        series_nam = series_name.split("_")
                        data.append({
                            'Series_Name': series_nam[1],
                            'Method_Name': method_name,
                            'Max_Depth': max_depth,
                            'Learning_Rate': learning_rate,
                            'Min_Data_In_Bin': min_data_in_bin,
                            'Min_Data_In_Leaf': min_data_in_leaf,
                            'Test MAPE': test_mape,
                            'Parameters': parameters_string.replace("\n", ",")
                        })

    return pd.DataFrame(data)

# Specify the root directory
root_directory = 'series'

# Process the directories and create the DataFrame
df = process_directories(root_directory)

# Display the DataFrame
print(df)

data=pd.read_csv(os.path.join("data", "filtered_short.csv"))
print(data)

merged = pd.merge(df, data, on='Series_Name')

print(merged)
merged.to_csv('summary.csv')
