#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:48:22 2025

@author: tanvibansal
"""

from IMM_MOT.models import KalmanFilter, IMM, MOT
from IMM_MOT.results import filter_results_to_df, sample_label_extract, event_label_extract
from glob import glob
import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


#select a month long sample to use as the target signal for the filter
sensor = "blue_eyed_tiger"
year = 2023
month = 9 

#read in measurements from selected month, tidy columns/rows
root_fp = "/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/"
data_fp = f"data-1015/{sensor}/"
fps = glob(root_fp + data_fp + f"{year}-{month:02d}-*.csv") 
measurements = pd.concat([pd.read_csv(fp) for fp in fps]).dropna(subset="depth_raw_mm")
measurements["time"] = pd.to_datetime(measurements["time"],format="ISO8601",utc=True)
measurements = measurements.sort_values(by="time")
measurements["t_elapsed"] = pd.to_numeric((measurements["time"] - measurements["time"].iloc[0]).dt.total_seconds())
measurements["dt"] = np.append([60],np.diff(measurements["t_elapsed"]))
measurements = measurements.loc[measurements.depth_raw_mm >= -30]

#define key model parameters
base_track_filters = [KalmanFilter(x=-10,H=np.array([[0.0,0.0,0.0]]),Q=np.array([[-20.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]),R = np.array([[30.0]]),uuid=000000),
                      KalmanFilter(x=-10,Q=np.array([[15.0,0.05,0.0],[10.0,0.04,0.0],[0.0,0.0,0.0]])/5,R = np.array([[40.0]]),uuid=000000)]
noise_track_Q = np.array([[10.0,0.05,0.0],[10.0,0.1,0.0],[0.0,0.0,0.0]])
noise_track_R = np.array([[15.0]])
gating_thresh = 10.0
track_expiration_thresh = 3/60

#run the MOT model with defined parameters
p,u,f,c = MOT(base_track_filters, noise_track_Q, noise_track_R, track_expiration_thresh, measurements, gating_thresh)

#tidy the model outputs into dataframes for evaluation
predictions_df, uncertainties_df, cost_df = filter_results_to_df(p,u,f,c)

#create dataframe with sample level predictions and labels
label_df = pd.read_csv("data/" + "sensor_events_uuid_tidy.csv")
sample_performance = sample_label_extract(measurements, f, label_df, sensor)

#create dataframe with aggregated event level predictions and labels
event_performance = event_label_extract(sample_performance)

#get sample and event level confusion matrices
sample_performance_eval = sample_performance.copy(deep=True).loc[sample_performance.depth_raw_mm >=15]
e_tn, e_fp, e_fn, e_tp = confusion_matrix(event_performance['true_label'], event_performance['pred_label'], labels=["noise","flood"]).ravel()
s_tn, s_fp, s_fn, s_tp = confusion_matrix(sample_performance_eval['true_label'], sample_performance_eval['pred_label'], labels=["noise","flood"]).ravel()

#plot filter prediction results
fig = plt.figure(figsize=(15, 6))
for i in predictions_df.columns:
    plt.fill_between(measurements.t_elapsed/60, predictions_df[i] - uncertainties_df[i], predictions_df[i] + uncertainties_df[i], alpha=0.5)
    plt.plot(measurements.t_elapsed/60,predictions_df[i], label=f'filter {i}', marker='.')
plt.plot(measurements.t_elapsed/60, measurements.depth_raw_mm, label='measured', color='mediumblue')
plt.ylim(-500, measurements.depth_raw_mm.max() + 100)
plt.title(f"Filter Results for {sensor}, month: {month}, year: {year}")
plt.legend(ncol=3,loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

#plot sample level and event level confusion matrices
cnfsnfig,ax = plt.subplots(ncols=2,figsize=(12,5))
cnfsnfig.suptitle(f"Confusion Matrix for {sensor}, month: {month}, year: {year}",fontsize="x-large")
sns.heatmap(np.array([[e_tp,e_fn],[e_fp,e_tn]]), annot=True, fmt='d', ax=ax[0], cmap='Blues', xticklabels=["flood","noise"], yticklabels=["flood","noise"])
ax[0].set_xlabel("Predicted Label")
ax[0].set_ylabel("Actual Label")
ax[0].set_title("Event Level Confusion Matrix")

sns.heatmap(np.array([[s_tp,s_fn],[s_fp,s_tn]]), annot=True, fmt='d',ax=ax[1], cmap='Greens', xticklabels=["flood","noise"], yticklabels=["flood","noise"])
ax[1].set_xlabel("Predicted Label")
ax[1].set_ylabel("Actual Label")
ax[1].set_title("Sample Level Confusion Matrix")

#%% develop evaluation here 
from IMM_MOT.evaluation import evaluate_results
label_mappings={
    "flood":"flood",
    "noise":"noise",
    "blip":"blip",
    "box":"noise",
    "misc":"noise",
    "pulse-chain":"noise"
    }
event_results_table, recall, precision, specificity = evaluate_results(event_performance,label_mappings=label_mappings)
sample_results_table, sample_recall, sample_precision, sample_specificity = evaluate_results(sample_performance_eval,label_mappings=label_mappings)







