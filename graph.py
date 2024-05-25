#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ownjo
"""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np

### TUNING AND CONFIG
data_loc = os.path.join("data","timeseries.csv")
pdf_loc = os.path.join("series", "graphs.pdf")

### LOAD DATA 
data = pd.read_csv(data_loc)

### GRAPH
pdf = PdfPages(pdf_loc)

for index,row in data.iterrows():
    # Вземаме ред с данни
    series_name = "%s_%s"% (row["Competition"], row["Series_Name"])
    series_category = row["Category"]
    series_type = row["Type"]
    seasonality = row["Seasonality"]
    number_of_observations = row["Number_Of_Observations"]
    number_of_predictions = row["Number_Of_Predictions"]
    total_datapoints = row["Total_Datapoints"]
    data=row[8:].dropna().to_list()
#    print(data)

    fig = plt.figure(index, figsize=(28,15))
    plt.grid(True, dashes=(1,1))
    plt.title(series_name+"  "+" "+series_category+" "+series_type)
    plt.xticks(rotation=90)
    plt.plot(data, color="blue", label="Original data")
    plt.axvline(x=number_of_observations, color="red", linestyle="--", label="Forecast horizon")
    plt.legend()
#    plt.show()
    png_dir = os.path.join("series", series_name)
    os.makedirs(png_dir, exist_ok=True)
    png_path = os.path.join("series", series_name, "graph.png")
    plt.savefig(png_path)
    pdf.savefig(fig)
    plt.close(fig)
    print(index, file=sys.stderr)

pdf.close()
