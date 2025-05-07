#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 02:24:38 2025

@author: tanvibansal
"""

from performance_measures import get_confusion_label, performance_metrics, events_result_table
from LSTM.results import plot_confusion_matrix_with_metrics, predictions_to_df
from sklearn.metrics import confusion_matrix


def get_performance_metrics(test_results):
    tn, fp, fn, tp = confusion_matrix(test_results["class"], test_results["predicted"], labels=["noise","flood"]).ravel()
    recall, precision, specificity = performance_metrics(tn, fp, fn, tp)
    return recall, precision, specificity

def evaluate_results(test_ds, test_prob, test_pred, test_uuids,label_mappings=None):
    test_results = predictions_to_df(test_ds, test_prob, test_pred, test_uuids).dropna()
    test_results["cnfsn"] = test_results.apply(lambda x: get_confusion_label(x.predicted, x["class"]),axis=1)
    event_results_table = events_result_table(test_results,label_mappings=label_mappings)
    recall, precision, specificity = get_performance_metrics(test_results)
    return event_results_table, recall, precision, specificity
