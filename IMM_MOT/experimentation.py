#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:03:17 2025

@author: tanvibansal
"""

from IMM_MOT.models import MOT, KalmanFilter, IMM
from IMM_MOT.results import filter_results_to_df, sample_label_extract, event_label_extract
from matplotlib import pyplot as plt
import os
import pandas as pd
from glob import glob
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random

root_fp = "data/"
label_df = pd.read_csv(root_fp + "sensor_events_uuid_tidy.csv")
train,test,val = [pd.read_csv(f"{root_fp}{i}.csv") for i in ["train","test","val"]]

#%% randomly generate 2 week event boundaries around each event; event will be randomly placed within the window
test_exp = test.copy(deep=True)
#two-week sliding window around event of interest, randomly select prior and posterior buffers
x = test.iloc[0]
window_seconds=1*1*24*60*60
def get_random_window(x,window_seconds):
    start,end = pd.to_datetime([x.start_time, x.end_time],format='ISO8601')
    window = pd.Timedelta(seconds=window_seconds) #2 weeks in seconds 
    
    min_start = (end - window) # Calculate latest possible window start so event_end is still in window
    max_start = start # latest possible window start so event_end is still in window
    
    if min_start > max_start:
        window_start = start # event is longer than the window, fall back to window that just includes event_start
    else:
        delta_secs = (max_start - min_start).seconds
        random_offset = pd.Timedelta(seconds=random.randint(0, delta_secs))
        window_start = min_start + random_offset
        
    window_end = window_start + window
    return pd.Series({"window_start": window_start, "window_end": window_end})

test_exp = test_exp.join(test_exp.apply(lambda x: get_random_window(x,window_seconds),axis=1))

#%% run the filter over each event window in the test set and save predicted label/flood probability
base_track_filters = [KalmanFilter(x=-10,H=np.array([[0.0,0.0,0.0]]),Q=np.array([[-20.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]),R = np.array([[30.0]]),uuid=000000),
                      KalmanFilter(x=-10,Q=np.array([[15.0,0.05,0.0],[10.0,0.04,0.0],[0.0,0.0,0.0]])/5,R = np.array([[40.0]]),uuid=000000)]
noise_track_Q = np.array([[10.0,0.05,0.0],[10.0,0.1,0.0],[0.0,0.0,0.0]])
noise_track_R = np.array([[15.0]])
gating_thresh = 10.0
track_expiration_thresh = 3/60
x = test_exp.iloc[0]

def extract_window_measurements(test_exp):
    event_ts = []
    sensor_df_fp_parent = "/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/data-1015"
    sensors = test_exp.deployment_id.unique()
    for i in sensors:
        test_exp_i = test_exp.copy(deep=True).loc[test_exp.deployment_id == i].set_index("uuid")
        fps = glob(f"{sensor_df_fp_parent}/{i}/*.csv")
        if len(fps) > 0:
            #read in all the data .csvs for the sensor of interest and set the time as index
            sensor_df = pd.concat([pd.read_csv(fp) for fp in fps])
            sensor_df = sensor_df[["time","deployment_id","depth_raw_mm","depth_filt_stages_applied"]]
            sensor_df["time"] = pd.to_datetime(sensor_df["time"],format="ISO8601")#"%Y-%m-%d %H:%M:%S.%f+%z")
            sensor_df.set_index("time",inplace=True)
            sensor_df.sort_index(inplace=True)
            
            #start parsing each event
            def extract_event_measurements(x, sensor_df):
                #extract event timeseries in window bounds for each event
                x_df = sensor_df.loc[(sensor_df.index >= x.window_start) & (sensor_df.index <= x.window_end)]
                x_df = x_df.dropna(subset="depth_raw_mm") #drop nans
                x_df["t_elapsed"] = (x_df.index - x_df.index[0]).seconds
                x_df["dt"] = np.append([60],np.diff(x_df["t_elapsed"]))
                x_df = x_df.loc[x_df.depth_raw_mm >= -30]
                return x_df
            
            measurements_i = test_exp_i.apply(lambda x: extract_event_measurements(x,sensor_df),axis=1)
            event_ts.append(measurements_i)
        print("%s complete"%(i))
    return pd.concat(event_ts)

test_measurements = extract_window_measurements(test_exp)

def predict(measurements, base_track_filters, noise_track_Q, noise_track_R, gating_thresh, track_expiration_thresh): 
    #run the MOT model with defined parameters
    p,u,f,c = MOT(base_track_filters, noise_track_Q, noise_track_R, track_expiration_thresh, measurements, gating_thresh)
    return f 

test_filtered = []
for i in range(len(test_measurements)):
    x = test_measurements.iloc[i]
    f = predict(x, base_track_filters, noise_track_Q, noise_track_R, gating_thresh, track_expiration_thresh)
    test_filtered.append(f)
    if i%500 == 0:
        print(f"{i} complete")
        
pandarallel.initialize(nb_workers=8,progress_bar=True)
test_filtered = test_measurements.parallel_apply(lambda x: predict(x, base_track_filters, noise_track_Q, noise_track_R, gating_thresh, track_expiration_thresh))
    #tidy the model outputs into dataframes for plotting
    # predictions_df, uncertainties_df, cost_df = filter_results_to_df(p,u,f,c)
    
    #create dataframe with sample level predictions and labels
    sample_performance = sample_label_extract(measurements, f, test, sensor)
    
    #create dataframe with aggregated event level predictions and labels, select predicted label for our event of interest
    event_performance = event_label_extract(sample_performance)
    noise_prob, pred_label = event_performance.loc[event_performance.uuid == x.uuid,["noise_pct","pred_label"]].values.flatten()
    flood_prob = 1 - noise_prob

    return flood_prob
def get_predictions(x, base_track_filters, noise_track_Q, noise_track_R, gating_thresh, track_expiration_thresh):
    sensor = x.deployment_id
    
    window_start = x.window_start
    start_year = window_start.year
    start_month = window_start.month
    
    window_end = x.window_end
    end_year = window_end.year
    end_month = window_end.month

    root_fp = "/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/"
    data_fp = f"data-1015/{sensor}/"
    
    if f"{start_year}-{start_month:02d}" != f"{end_year}-{end_month:02d}":
        fps = glob(root_fp + data_fp + f"{start_year}-{start_month:02d}-*.csv") + glob(root_fp + data_fp + f"{end_year}-{end_month:02d}-*.csv")
    else:
        fps = glob(root_fp + data_fp + f"{start_year}-{start_month:02d}-*.csv") 

    measurements = pd.concat([pd.read_csv(fp) for fp in fps]).dropna(subset="depth_raw_mm")
    measurements["time"] = pd.to_datetime(measurements["time"],format="ISO8601",utc=True)
    measurements = measurements.sort_values(by="time")
    measurements = measurements.loc[(measurements.time >= window_start) & (measurements.time <= window_end)]
    measurements["t_elapsed"] = pd.to_numeric((measurements["time"] - measurements["time"].iloc[0]).dt.total_seconds())
    measurements["dt"] = np.append([60],np.diff(measurements["t_elapsed"]))
    measurements = measurements.loc[measurements.depth_raw_mm >= -30]    

    #run the MOT model with defined parameters
    p,u,f,c = MOT(base_track_filters, noise_track_Q, noise_track_R, track_expiration_thresh, measurements, gating_thresh)
    
    #tidy the model outputs into dataframes for plotting
    # predictions_df, uncertainties_df, cost_df = filter_results_to_df(p,u,f,c)
    
    #create dataframe with sample level predictions and labels
    sample_performance = sample_label_extract(measurements, f, test, sensor)
    
    #create dataframe with aggregated event level predictions and labels, select predicted label for our event of interest
    event_performance = event_label_extract(sample_performance)
    noise_prob, pred_label = event_performance.loc[event_performance.uuid == x.uuid,["noise_pct","pred_label"]].values.flatten()
    flood_prob = 1 - noise_prob

    return flood_prob


#%%
from pandarallel import pandarallel 
pandarallel.initialize(nb_workers=7,progress_bar=True)

meas =  test_exp.parallel_apply(lambda x: get_window_measurements(x),axis=1)
meas = meas.to_frame().rename(columns={0:"measurements"}).merge(test_exp["deployment_id"],left_index=True,right_index=True)

probs = meas.parallel_apply(lambda x: predict(x.deployment_id, x.measurements, base_track_filters, noise_track_Q, noise_track_R, gating_thresh, track_expiration_thresh),axis=1)

flood_probs = test_exp[0:5000].parallel_apply(lambda x: get_predictions(x, base_track_filters, noise_track_Q, noise_track_R, gating_thresh, track_expiration_thresh),axis=1)
# flood_probs = []
# for i in range(len(test_exp)):
#     x = test_exp.iloc[i]
#     pr = get_predictions(x, base_track_filters, noise_track_Q, noise_track_R, gating_thresh, track_expiration_thresh)
#     flood_probs.append(pr)
#     if i % 1000 == 0:
#         print(f"{i} complete")











