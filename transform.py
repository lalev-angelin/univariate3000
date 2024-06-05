#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

### TUNING & CONFIG
nn1_data_loc = os.path.join("data", "original", "nn1.csv")
m1_data_loc = os.path.join("data", "original", "m1.csv")
m3_data_loc = os.path.join("data", "original", "m3.csv")
final_data_loc = os.path.join("data","timeseries.csv")
filtered_data_loc = os.path.join("data", "filtered.csv")
filtered_short_loc = os.path.join("data", "filtered_short.csv")


### IMPORT & TRANSFORM OF M1 DATASET

m1_data=pd.read_csv(m1_data_loc, header=0)
m1_data["Competition"]="M1"

m1_data.rename(columns={"Series":"Series_Name", 
    "N Obs":"Number_Of_Observations",
    "NF":"Number_Of_Predictions",
    "Number Of Subperiods": "Number_Of_Subperiods",
    "Starting Subperiod": "Starting_Subperiod",
    "Starting date": "Starting_Date"}, inplace=True)

m1_data.dropna(axis=1, inplace=True, how='all')
m1_data["Total_Datapoints"]=m1_data['Number_Of_Observations']+m1_data['Number_Of_Predictions']

# Reorder m1 dataset columns so they match the ones of nn1 
# dataset that will be concatenated later. 

col_list = m1_data.columns.to_list()
#print(col_list)

# Send "Competition" column to the first place
val = col_list.pop(col_list.index("Competition"))
col_list.insert(0, val)
# Send "Series Name" column to the second place
val = col_list.pop(col_list.index("Series_Name"))
col_list.insert(1, val)
# Send "Category" column to the third place
val = col_list.pop(col_list.index("Category"))
col_list.insert(2, val)
# Send "Type" column to the fourth place
val = col_list.pop(col_list.index("Type"))
col_list.insert(3, val)
# Send "Type" column to the fifth place
val = col_list.pop(col_list.index("Seasonality"))
col_list.insert(4, val)
# Send Number_Of_Observations sixth 
val = col_list.pop(col_list.index("Number_Of_Observations"))
col_list.insert(5, val)
# Send Number_Of_Predictions seventh 
val = col_list.pop(col_list.index("Number_Of_Predictions"))
col_list.insert(6, val)
# Send Total_Datapoints eighth 
val = col_list.pop(col_list.index("Total_Datapoints"))
col_list.insert(7, val)
# Send Number_Of_Subperiods to ninth place
val = col_list.pop(col_list.index("Number_Of_Subperiods"))
col_list.insert(8, val)
# Send Starting_Subperiod to tenth place
val = col_list.pop(col_list.index("Starting_Subperiod"))
col_list.insert(9, val)
# Send Starting_Date to eleventh place
val = col_list.pop(col_list.index("Starting_Date"))
col_list.insert(10, val)

#print(col_list)
#print(m1_data.head)
#sys.exit(1)

m1_data=m1_data[col_list]
#print(m1_data.head)
#sys.exit(1)
#m1_data.to_csv("dump1.csv")
#sys.exit(1)


### IMPORT & TRANSFORM OF NN1 DATASET
nn_data=pd.read_csv(nn1_data_loc, header=None)
columns = nn_data.shape[1]
names = ["No", "Competition", "Group", "Series_Name", "Type",
        "Total_Datapoints", "Number_Of_Predictions"]
start = len(names)
for i in range(1, columns-start+1): 
    names +=["%i"%i] 
nn_data.columns=names

nn_data.drop(["Group","No"], axis=1, inplace=True)
nn_data['Type']=nn_data['Type'].str.upper()

nn_data['Number_Of_Subperiods']=-1
nn_data['Starting_Subperiod']=-1
nn_data['Starting_Date']=""

nn_data.insert(2, "Category", "OTHER")
nn_data.insert(4, "Seasonality", pd.NA) 
nn_data.insert(6, 'Number_Of_Observations', nn_data['Total_Datapoints']-nn_data['Number_Of_Predictions'])

col = nn_data.pop('Total_Datapoints')
nn_data.insert(7, 'Total_Datapoints', col)

col = nn_data.pop('Number_Of_Subperiods')
nn_data.insert(8, 'Number_Of_Subperiods', col)

col = nn_data.pop('Starting_Subperiod')
nn_data.insert(9, 'Starting_Subperiod', col)

col = nn_data.pop('Starting_Date')
nn_data.insert(10, 'Starting_Date', col)

#print(nn_data.head)
#sys.exit(1)
nn_data.to_csv("dump2.csv")
#sys.exit(1)

### IMPORT & TRANSFORM OF M3 DATASET
m3_data=pd.read_csv(m3_data_loc, header=0)
m3_data["Competition"]="M3"
m3_data.rename(columns={"Series":"Series_Name", 
    "N":"Total_Datapoints",
    "NF":"Number_Of_Predictions",
    "Number Of Subperiods": "Number_Of_Subperiods",
    "Starting Month": "Starting_Subperiod", 
    "Starting Year": "Starting_Date"}, inplace=True)

m3_data["Number_Of_Observations"]=m3_data['Total_Datapoints']-m3_data['Number_Of_Predictions']
m3_data.insert(2, "Seasonality", pd.NA) 

# Reorder m1 dataset columns so they match the ones of nn1 
# dataset that will be concatenated later. 

col_list = m3_data.columns.to_list()
#print(col_list)

# Send "Competition" column to the first place
val = col_list.pop(col_list.index("Competition"))
col_list.insert(0, val)
# Send "Series Name" column to the second place
val = col_list.pop(col_list.index("Series_Name"))
col_list.insert(1, val)
# Send "Category" column to the third place
val = col_list.pop(col_list.index("Category"))
col_list.insert(2, val)
# Send "Type" column to the fourth place
val = col_list.pop(col_list.index("Type"))
col_list.insert(3, val)
# Send "Type" column to the fifth place
val = col_list.pop(col_list.index("Seasonality"))
col_list.insert(4, val)
# Send Number_Of_Observations sixth 
val = col_list.pop(col_list.index("Number_Of_Observations"))
col_list.insert(5, val)
# Send Number_Of_Predictions seventh 
val = col_list.pop(col_list.index("Number_Of_Predictions"))
col_list.insert(6, val)
# Send Total_Datapoints eighth 
val = col_list.pop(col_list.index("Total_Datapoints"))
col_list.insert(7, val)
# Send Number_Of_Subperiods to ninth place
val = col_list.pop(col_list.index("Number_Of_Subperiods"))
col_list.insert(8, val)
# Send Starting_Subperiod to tenth place
val = col_list.pop(col_list.index("Starting_Subperiod"))
col_list.insert(9, val)
# Send Starting_Date to eleventh place
val = col_list.pop(col_list.index("Starting_Date"))
col_list.insert(10, val)
#print(col_list)
#print(m1_data.head)
#sys.exit(1)

m3_data=m3_data[col_list]
#print(m3_data.head)
#sys.exit(1)
m3_data.to_csv("dump3.csv")
#sys.exit(1)


### FINAL CONCATENATION
# Careful! We are lazy and the row below will work only 
# if the frames are in descending order by the number of columns.
final = pd.concat([nn_data, m1_data, m3_data], axis=0, ignore_index=True) 
#print(final.columns)
#print(final.head)

# Removing the dots and changing dashes to underlines to accomodate
# some R processing by team members
final['Series_Name']=final['Series_Name'].str.replace(".","")
final['Series_Name']=final['Series_Name'].str.replace("-","_")
final['Series_Name']=final['Series_Name'].str.replace(" ","")

### "CATCH-ALL" FIXES
 
# Remove whitespace after the strings in all columns
final = final.applymap(lambda x: x.strip() if isinstance(x, str) else x)
final['Type']=final['Type'].replace({"MONTHLYLY":"MONTHLY"})

# Save
final.to_csv(final_data_loc, index=False)

final = final[final['Type'].isin(['YEARLY', 'MONTHLY', 'QUARTERLY'])]
final.to_csv(filtered_data_loc, index=False)

final = final[final['Total_Datapoints']>30]
final.to_csv(filtered_short_loc, index=False)
