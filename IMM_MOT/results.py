#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:03:01 2025

@author: tanvibansal
"""

import numpy as np 
import pandas as pd

def filter_results_to_df(predictions,uncertainties,filters,costs):
    """
    function to intake nested list arguments of variable shape due to track generation/expiration and convert to tabular data format 
    Args: 
        predictions: nested list of predicted locations for each object
        uncertainties: nested list of standard deviation of prediction for each object
        filters: list containing filter ID's for each object/track generated in algorith
        costs: nested list containing measurement/prediction costs for each object
    """
    f_flat = np.unique(np.array([item for sublist in filters for item in sublist]))
    predictions_df = {f"{key}": [] for key in f_flat}
    uncertainties_df = {f"{key}": [] for key in f_flat}
    cost_df = {f"{key}": [] for key in f_flat}
    
    predictions_df["-1"]=[]
    uncertainties_df["-1"]=[]
    #cost_df["-1"]=[]
    
    for i in range(len(filters)):
        #find active filters at time t
        filters_i = np.array(filters[i])
        filters_inactive = f_flat[~np.isin(f_flat,filters_i)]
        #append predictions from rest and then all active filters
        predictions_df["-1"].append(predictions[i][0])
        uncertainties_df["-1"].append(uncertainties[i][0])
        #cost_df["-1"].append(costs[i][0])
        
        for j in range(1,len(filters_i)):
            filter_j = filters_i[j]
            predictions_df[f"{filter_j}"].append(predictions[i][j])
            uncertainties_df[f"{filter_j}"].append(uncertainties[i][j])
           # cost_df[f"{filter_j}"].append(costs[i][j])
        #append nans from inactive filters 
        for k in range(len(filters_inactive)):
            filter_j = filters_inactive[k]
            predictions_df[f"{filter_j}"].append(np.nan)
            uncertainties_df[f"{filter_j}"].append(np.nan)
           # cost_df[f"{filter_j}"].append(np.nan)
            
    predictions_df = pd.DataFrame(predictions_df)
    uncertainties_df = pd.DataFrame(uncertainties_df)
    cost_df = pd.DataFrame(cost_df)
    return predictions_df, uncertainties_df , cost_df


def sample_label_extract(measurements, f, label_df, sensor):
    """
    function to intake measurement data frame, predicted track IDs, and label dataframe and return a dataframe 
    containing predicted and actual labels for each record in the measurement df
    arguments: 
        measurements: pandas dataframe of continuous timeseries signals with columns: "time","deployment_id","depth_raw_mm","depth_proc_mm",
            "depth_filt_stages_applied","t_elapsed","dt"
        f: nested list with same length as measurements containing predicted active tracks for each sample
        label_df: reference label dataframe
        sensor: name of the sensor (deployment_id) of the target measurements
        
    """
    #copy measurement dataframe 
    sample_performance = measurements.copy(deep=True).set_index("time")#.loc[:,["time","deployment_id","depth_raw_mm","depth_proc_mm","depth_filt_stages_applied","t_elapsed","dt"]].set_index("time")
    sample_performance.sort_index(inplace=True)
    
    #assign predicted label for each sample from filter IDs
    #loop through active filter IDs for each sample in the df, assign label "flood" if all active filters IDs are 0 (base track), "noise" if nonzero filter IDs are active
    sample_performance["pred_label"] = ["flood" if np.all(np.array(i) == 0) else "noise" for i in f]
    sample_performance.loc[sample_performance.depth_raw_mm <=30, "pred_label"] = "rest" #change sample label to rest if depth_raw_mm <= 30
    
    #extract/reference actual label for each sample from labelled dataframe
    sample_performance["event_class"] = "rest"
    sample_performance["max_depth"] = 0
    def extract_event_labels(x,sensor_df): #cross reference label dataframe with measurement dataframe to extract actual labels directly in sample_performance df
                   sensor_df.loc[(sensor_df.index >= x.start_time) & (sensor_df.index <= x.end_time),"event_class"] = x.label
                   sensor_df.loc[(sensor_df.index >= x.start_time) & (sensor_df.index <= x.end_time),"uuid"] = x.uuid
                   sensor_df.loc[(sensor_df.index >= x.start_time) & (sensor_df.index <= x.end_time),"max_depth"] = sensor_df.loc[(sensor_df.index >= x.start_time) & (sensor_df.index <= x.end_time),"depth_raw_mm"].max()
    
    label_df.loc[label_df.deployment_id == sensor].apply(lambda x: extract_event_labels(x, sample_performance),axis=1) 
    
    #create "true_label" column and flip all nonflood labels to "noise" 
    sample_performance["true_label"] = sample_performance["event_class"]
    sample_performance.loc[~np.isin(sample_performance["event_class"], ["rest","flood"]),"true_label"] = "noise"
    return sample_performance


def event_label_extract(sample_performance):
    """
    function to intake the measurement dataframe with columns for the predicted and actual labels and return aggregated event level labels

    """
    event_performance = sample_performance.copy(deep=True).loc[(sample_performance.depth_raw_mm >= 10) & (sample_performance.true_label != "rest")] #drop measurements where depth_filt_mm = 0
    
    event_performance = event_performance.groupby(['uuid','deployment_id','event_class','true_label']).agg(
        max_depth = ('max_depth','mean'),
        num_points=('pred_label', 'size'),
        flood_pct=('pred_label', lambda x: (x == 'flood').mean()),
        noise_pct=('pred_label', lambda x: (x == 'noise').mean()),
        noise_pct_h = ('depth_filt_stages_applied', 'count')
    ).reset_index()
    #imm filter predicted label 
    event_performance["pred_label"] = event_performance.apply(lambda x: "noise" if x["noise_pct"] > 0.5 else "flood",axis=1)#["flood","noise"][np.argmax(x[["flood_pct","noise_pct"]])],axis=1)
    #heur filter predicted label
    event_performance["noise_pct_h"] = event_performance["noise_pct_h"]/event_performance["num_points"]
    event_performance["flood_pct_h"] = 1 - event_performance["noise_pct_h"]
    event_performance["pred_label_h"] = event_performance.apply(lambda x: ["flood","noise"][np.argmax(x[["flood_pct_h","noise_pct_h"]])],axis=1)

    return event_performance
