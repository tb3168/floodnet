#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 01:51:35 2025

@author: tanvibansal
"""

from plotnine import *
from utils.depth_segment_finder import depth_segment_finder
import pandas as pd

def get_false_negative_df(false_neg_ids, ds):
    false_neg_df = ds.loc[false_neg_ids]
    false_neg_signal = false_neg_df["signal"].apply(pd.Series)
    false_neg_signal["t_max"] = false_neg_signal.apply(lambda x: x.time[-1] ,axis=1)
    false_neg_signal[["k","depth_k"]] = false_neg_df.signal_k_partial.apply(lambda x: pd.Series({"k": x["time"][-1], "depth_k": x["depth_raw_mm"][-1]}))
    false_neg_signal["depth_segment"] = false_neg_signal.depth_raw_mm.apply(lambda x: depth_segment_finder(max(x)))
    false_neg_signal = false_neg_signal.explode(["time","depth_raw_mm"]).astype({'time': 'int', 'depth_raw_mm': 'float64'})
    false_neg_signal["t_pct"] = false_neg_signal["time"]/false_neg_signal["t_max"]
    false_neg_signal["k"] = false_neg_signal["k"]/false_neg_signal["t_max"]
    false_neg_signal = false_neg_signal.reset_index()
    
    p = ggplot(false_neg_signal,aes(x="t_pct",y = "depth_raw_mm",group="uuid")) + geom_line(alpha=0.5) + \
    geom_point(aes(x="k",y="depth_k",group="uuid")) + ylim(-100,false_neg_signal.depth_raw_mm.max()+15) + \
    labs(title="False Negatives for LSTM Classifier")
    p.show()
    
    return false_neg_signal





