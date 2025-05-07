#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:12:08 2025

@author: tanvibansal
"""

import numpy as np 
import pandas as pd 

def get_confusion_label(pred_label, true_label):
    if pred_label == "flood" and true_label == "flood":
        cnfsn = "TP"
    if pred_label == "flood" and true_label != "flood":
        cnfsn = "FP"
    if pred_label != "flood" and true_label == "flood":
        cnfsn = "FN"
    if pred_label != "flood" and true_label != "flood":
        cnfsn = "TN"
    return cnfsn


def performance_metrics(tn,fp,fn,tp):
    if tp + fn == 0:
        recall = np.nan
    else:
        recall = tp/(tp+fn)
    
    if tp+fp == 0:
        precision = np.nan
    else:
        precision = tp/(tp+fp)
    
    if tn+fp == 0:
        specificity = np.nan
    else:
        specificity = tn/(tn+fp)
    
    return recall, precision, specificity 


def events_result_table(test_results, label_mappings=None):
    """
    test_results: dataframe with column "label" for true label, "cnfsn" for confusion
    label_mappings: dictionary with mappings to transform labels to for output events table result 

    """
    if label_mappings:
        new_labels = [label_mappings[test_results.label.iloc[i]] for i in range(len(test_results))]
        test_results["label"] = new_labels
    
    event_label_results = test_results.groupby("label").agg(
        num_events=('label','count'),
        num_nonfloods=('cnfsn', lambda x: (x == 'TN').sum() + (x == 'FN').sum())
        )
    event_label_results["nonflood_pct"] = round(event_label_results["num_nonfloods"]*100/event_label_results["num_events"],2)
    return event_label_results