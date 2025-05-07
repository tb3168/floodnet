#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:29:06 2025

@author: tanvibansal
"""

from performance_measures import get_confusion_label, performance_metrics, events_result_table
from sklearn.metrics import confusion_matrix


def get_performance_metrics(event_performance):
    tn, fp, fn, tp = confusion_matrix(event_performance["label"], event_performance["pred_label"], labels=["noise","flood"]).ravel()
    recall, precision, specificity = performance_metrics(tn, fp, fn, tp)
    return recall, precision, specificity

def evaluate_results(test_ds, pflood_test, pred_test, label_mappings=None):
    event_performance = test_ds.copy(deep=True).dropna().drop(columns=["signal","signal_k_partial"])
    event_performance["proba"] = pflood_test
    event_performance["pred_label"] = ["flood" if i == 1 else "noise" for i in pred_test]

    event_performance["cnfsn"] = event_performance.apply(lambda x: get_confusion_label(x["pred_label"],x["label"]),axis=1)
    event_results_table = events_result_table(event_performance,label_mappings=label_mappings)
    recall, precision, specificity = get_performance_metrics(event_performance)
    return event_results_table, recall, precision, specificity
