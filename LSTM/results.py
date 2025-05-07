#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 21:53:32 2025

@author: tanvibansal
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd 

def plot_confusion_matrix_with_metrics(y_true, y_pred, test_prec, test_rec, class_mapping):
    class_vals = [i for i in class_mapping.keys()]
    class_names = [class_mapping[i] for i in range(len(class_vals))]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_vals)))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float)   # row-wise percentage (sensitivity)

    # Calculate Sensitivity (Recall) and PPV (Precision)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    ppv = np.diag(cm) / np.sum(cm, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax,annot_kws={"size": 20})
    
    # Labels inside the boxes
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            #percent = cm_perc[i, j]
            cell_value = cm_perc[i, j]
            text_color = "white" if cell_value > 0.9 else "black"
            ax.text(j + 0.5, i + 0.65, f"{count:,}", ha="center", va="center", color=text_color, fontsize=19)

    # Set axis labels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('Actual Label', fontsize=16)
    ax.set_xticks(np.arange(len(class_vals)) + 0.5)
    ax.set_yticks(np.arange(len(class_vals)) + 0.5)
    ax.set_xticklabels(class_names, rotation=0, ha="center",fontsize=14)
    ax.set_yticklabels(class_names, rotation=0,fontsize=14)

    # Plot PPV (precision) on top
    for j, p in enumerate(ppv):
        ax.text(j + 0.5, -0.1, f"{int(p*100) if not np.isnan(p) else 0}%", 
                ha='center', va='center', color='black', fontsize=12)

    # Plot Sensitivity (recall) on right
    for i, s in enumerate(sensitivity):
        ax.text(len(class_names) + 0.1, i + 0.5, f"{int(s*100) if not np.isnan(s) else 0}%", 
                ha='left', va='center', color='black', fontsize=12)

    #plot precision and recall on bottom
    ax.text(1.0, 2.4, f"precision: {test_prec:.2f}, recall: {test_rec:.2f} ", ha='center', va='center', color='dimgrey', fontsize=15)

    # Center "PPV" label on top
    ax.text(len(class_vals) / 2, -0.2, "PPV", fontsize=14, ha='center', va='center')

    # Center "Sensitivity" label on the right
    ax.text(len(class_vals) + 0.45, len(class_vals) / 2, "Sensitivity", fontsize=13, ha='center', va='center', rotation=90)

    plt.tight_layout()
    plt.show()
    
def predictions_to_df(test_ds, test_prob, test_pred, test_uuids):
    """
    function to tidy prediction results from LSTM classifier into dataframe to prep for evaluation
    arguments:
        test_ds: dataframe for test set with columns signal, signal_k_partial, and index uuid
        test_prob: logits from LSTM classifier
        test_pred: class predictions from LSTM classifier
        test_uuids: uuids of entries in order from LSTM classifier
    """
    test_performance = test_ds.copy(deep=True).drop(columns=["signal","signal_k_partial"])
    proba = torch.nn.functional.sigmoid(test_prob[:,0])
    pred = ["flood" if p == 1 else "noise" for p in test_pred]
    test_results = pd.DataFrame(data={"predicted":pred,"proba":proba,"uuid":test_uuids}).set_index("uuid")
    test_performance = test_performance.join(test_results)
    return test_performance
