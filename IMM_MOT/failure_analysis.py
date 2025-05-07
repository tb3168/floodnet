#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:10:08 2025

@author: tanvibansal
"""
import numpy as np 
import pandas as pd
from glob import glob
from IMM_MOT.models import KalmanFilter, IMM, MOT
from IMM_MOT.results import filter_results_to_df, sample_label_extract, event_label_extract
from matplotlib import pyplot as plt 

root_fp = "/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/"

def plot_false_negatives(event_performance, label_df):
    """
    function to take in event_performance dataframe and label dataframe, extract original timeseries, and plot
    arguments:
        event_performance: dataframe with event level aggregate labels and columns "cnfsn" and "uuid"
        label_df: labelled reference dataframe with columns "uuid"

    """
    event_fn_df = event_performance.loc[event_performance.cnfsn == "FN"]
    label_fn_df = label_df.merge(event_fn_df,on="uuid")

    #loop through each false negative record and plot
    for i in range(len(label_fn_df)):
        sensor = label_fn_df.deployment_id.iloc[i]
        sy = pd.to_datetime(label_fn_df.start_time.iloc[i]).year
        sm = "%02d"%(pd.to_datetime(label_fn_df.start_time.iloc[i]).month)
        sd = "%02d"%(pd.to_datetime(label_fn_df.start_time.iloc[i]).day)
        
        ey = pd.to_datetime(label_fn_df.end_time.iloc[i]).year
        em = "%02d"%(pd.to_datetime(label_fn_df.end_time.iloc[i]).month)
        ed = "%02d"%(pd.to_datetime(label_fn_df.end_time.iloc[i]).day)
        
        dpm = label_fn_df.max_depth_proc_mm.iloc[i]
        data_fp = f"data-1015/{sensor}/"
        fps = glob(root_fp + data_fp + f"{sy}-{sm}-{sd}.csv") + glob(root_fp + data_fp + f"{sy}-{sm}-{ed}.csv")
        fps = np.unique(np.array(fps))
        
        
        measurements = pd.concat([pd.read_csv(fp) for fp in fps]).dropna(subset="depth_raw_mm")
        measurements["time"] = pd.to_datetime(measurements["time"],format="ISO8601",utc=True)
        measurements = measurements.sort_values(by="time")
        measurements["t_elapsed"] = pd.to_numeric((measurements["time"] - measurements["time"].iloc[0]).dt.total_seconds())
        measurements["dt"] = np.append([60],np.diff(measurements["t_elapsed"]))
        
        #filter rows where depth_raw_mm <= -30mm 
        measurements = measurements.loc[measurements.depth_raw_mm >= -30]
        
        #run the MOT model
        base_track_filters = [KalmanFilter(x=-10,H=np.array([[0.0,0.0,0.0]]),Q=np.array([[-20.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]),R = np.array([[30.0]]),uuid=000000),
                              KalmanFilter(x=-10,Q=np.array([[15.0,0.05,0.0],[10.0,0.04,0.0],[0.0,0.0,0.0]])/5,R = np.array([[40.0]]),uuid=000000)]
        noise_track_Q = np.array([[10.0,0.05,0.0],[10.0,0.1,0.0],[0.0,0.0,0.0]])
        noise_track_R = np.array([[15.0]])
        gating_thresh = 10.0
        track_expiration_thresh = 3/60
    
        #run the MOT model with defined parameters
        p,u,f,c = MOT(base_track_filters, noise_track_Q, noise_track_R, track_expiration_thresh, measurements, gating_thresh)
        
        #tidy the results into a dataframe
        predictions_df, uncertainties_df, cost_df = filter_results_to_df(p,u,f,c)
        
        #plot the filter prediction results and save to disk
        fig = plt.figure(figsize=(15, 6))
        for i in predictions_df.columns:
            plt.fill_between(measurements.t_elapsed/60, predictions_df[i] - uncertainties_df[i], predictions_df[i] + uncertainties_df[i], alpha=0.5)
            plt.plot(measurements.t_elapsed/60,predictions_df[i], label=f'filter {i}', marker='.')
        plt.plot(measurements.t_elapsed/60, measurements.depth_raw_mm, label='measured', color='mediumblue')
        plt.ylim(-100, dpm + 100)
        plt.title(f"Filter Results for {sensor}, month: {sm}, year: {sy}")
        plt.legend(ncol=3,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
        
        #fig.savefig(path + f"{sensor}_{month}_{year}_filter_results_sample.png")
        #plt.close()
        #print(f"{sensor} predictions complete, tidied, and plotted")