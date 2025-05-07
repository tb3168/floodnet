#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:15:27 2025

@author: tanvibansal
"""

from matplotlib import pyplot as plt
import os
import pandas as pd
from glob import glob
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
from LSTM.datasets import *
from FFT_Feature_Classifier.models import Transform, get_spectral_features, get_cepstrum_features
from sklearn.ensemble import RandomForestClassifier

root_fp = "data/"
label_df = pd.read_csv(root_fp + "sensor_events_uuid_tidy.csv")
train,test,val = [pd.read_csv(f"{root_fp}{i}.csv") for i in ["train","test","val"]]

#%% load parent data sets and super parameters 
buffer = 0 
k_vals = [18]
train_ds = dataLoader(train, buffer)
train_ds.load_data()
train_ds.augment_floods(50)
train_ds = train_ds.k_partial_data(k_vals)

val_ds = dataLoader(val, buffer)
val_ds.load_data()
val_ds = val_ds.k_partial_data(k_vals)

test_ds = dataLoader(test, buffer)
test_ds.load_data()
test_ds = test_ds.k_partial_data(k_vals)

#%% run transformation steps on train, test, val sets 
k=18
do_reflect=True
do_normalize=True
order=1
transform = Transform(train_ds, k, do_reflect, do_normalize, order)
fft_train, ceps_train, labels_train, uuids_train = transform.transform()

transform = Transform(val_ds, k, do_reflect, do_normalize, order)
fft_val, ceps_val, labels_val, uuids_val = transform.transform()

transform = Transform(test_ds, k, do_reflect, do_normalize, order)
fft_test, ceps_test, labels_test, uuids_test = transform.transform()

#%% run feature extraction on train, test, val sets

feats_train = fft_train.apply(lambda x: get_spectral_features(x.freqs, x.amplitude),axis=1).\
    join(ceps_train.apply(lambda x: get_cepstrum_features(x.freqs, x.amplitude),axis=1)).fillna(0.0)

feats_val = fft_val.apply(lambda x: get_spectral_features(x.freqs, x.amplitude),axis=1).\
    join(ceps_val.apply(lambda x: get_cepstrum_features(x.freqs, x.amplitude),axis=1)).fillna(0.0)
    
feats_test = fft_test.apply(lambda x: get_spectral_features(x.freqs, x.amplitude),axis=1).\
    join(ceps_test.apply(lambda x: get_cepstrum_features(x.freqs, x.amplitude),axis=1)).fillna(0.0)
    
#%% train and test classifier
pos_weight = 6.0 
clf = RandomForestClassifier(max_depth=10, random_state=29, min_samples_leaf =4, max_features = "log2", \
                             bootstrap=True, n_jobs = -1, class_weight = {1:pos_weight, 0:1.0})
clf.fit(feats_train,labels_train)

#test on val set
flood_prob = clf.predict_proba(feats_val)[:,np.argwhere(clf.classes_ == 1).item()]
pred = clf.predict(feats_val)


#%% evaluate
from FFT_Feature_Classifier.evaluation import evaluate_results
pflood_test = clf.predict_proba(feats_test)[:,np.argwhere(clf.classes_ == 1).item()]
pred_test = clf.predict(feats_test)

label_mappings =  {
    "blip":"blip",
    "flood":"flood",
    "noise":"noise",
    "box":"noise",
    "misc":"noise",
    "pulse-chain":"noise"
    }

event_results_table, recall, precision, specificity = evaluate_results(test_ds, pflood_test, pred_test, label_mappings=label_mappings)
print(event_results_table.to_markdown())
print(precision, recall, specificity)

#test_performance.to_csv("FFT_test_predictions_k18.csv")
#%% failure analysis 
from FFT_Feature_Classifier.failure_analysis import get_false_negative_df
false_neg_ids = uuids_test[np.argwhere((labels_test != pred_test) & (labels_test == 1)).flatten()]
get_false_negative_df(false_neg_ids, test_ds)


