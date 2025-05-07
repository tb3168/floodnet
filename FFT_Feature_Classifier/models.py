#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:22:40 2025

@author: tanvibansal
"""

import numpy as np
from math import floor,ceil
from librosa import stft, fft_frequencies
from librosa.feature import *
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import rfft, rfftfreq

# =============================================================================
# TRANSFORM 
# =============================================================================
class Transform():
    def __init__(self, data, k, do_reflect, do_normalize, order, N=400):
        self.data = data.copy(deep=True).loc[data.k == k].dropna(subset = "signal_k_partial")
        self.reflect = do_reflect
        self.normalize = do_normalize 
        self.order = order
        self.N = N
        self.labels = (self.data["class"] == "flood").astype("int").values
        self.uuids = self.data.index.values
    def transform(self):
        data = self.data.signal_k_partial.apply(pd.Series)
        if self.reflect:
            data = data.apply(lambda x: pd.Series(reflect(x.time, x.depth_raw_mm)),axis=1).rename(columns={0:"time",1:"depth"})
        if self.normalize: 
            data = data.apply(lambda x: pd.Series(normalize(x.time, x.depth)),axis=1).rename(columns={0:"time",1:"depth"})
        if self.order >= 0:
            data = data.apply(lambda x: ndiff(x.time, x.depth, 1),axis=1).rename({0:"diff"})
        data_fft = data.apply(lambda x: pd.Series(fourier(x,self.N))).rename(columns={0:"freqs",1:"amplitude"})
        data_ceps = data_fft.apply(lambda x: pd.Series(cepstrum(x.amplitude)),axis=1).rename(columns={0:"freqs",1:"amplitude"})
        return data_fft, data_ceps, self.labels, self.uuids
        
# reflect pad
def reflect(time, depth):
    depth_r = np.append(depth,np.flip(depth))
    t_final = time[-1] #extend timestamps to represent time deltas of input data in the reflected signal
    time = np.append(time, t_final + np.cumsum(np.flip(np.append(60,np.diff(time)))))
    return time, depth_r
    
# normalize 
def normalize(time, depth):
    maxd = np.max(depth) 
    if maxd != 0 and maxd is not None:
        depth_n = depth/maxd
    else:
        depth_n = depth
    return time, depth_n

#nth order deriv
def nth_diff(order, x, y):
    #print(order)
    if order == 0:
        out = y
        return(out)
    else:
        dydx = np.diff(y)/np.diff(x)
        x = x[1:]
        return(nth_diff(order - 1, x, dydx)) 
    
def ndiff(time, depth, order=1):
    return [nth_diff(o, time, depth) for o in range(order + 1)]

#fourier transform
def zero_pad(depth, N): 
    l = N - len(depth) 
    if l % 2 !=0:
        before = floor(l/2)
        after = ceil(l/2)
    else:
        before = int(l/2)
        after = int(l/2)
    ypad = np.pad(depth,(before,after),constant_values=0.0)
    return(ypad)

def fourier(depth_list, N=400):
    sr=1/60
    depths = [x[np.isfinite(x)] for x in depth_list]
    depth_padded = [zero_pad(d,N) for d in depths]
    fft = [np.abs(stft(dp,n_fft=N,win_length=N,center=False)) for dp in depth_padded]
    freqs = fft_frequencies(sr = sr, n_fft = N)
    return freqs, fft

#cepstrum transform 
def cepstrum(fft_list):
    sr=1/60
    ceps = [np.abs(stft(s.flatten(),n_fft=len(s),win_length=len(s),center=False)) for s in fft_list]
    freqs = fft_frequencies(sr = sr, n_fft = len(fft_list[0]))
    return freqs, ceps

# =============================================================================
# EXTRACT FEATURES
# =============================================================================

#spectral features 
def spectral_feature_compute(S,freqs,sr=1/60,roll_percent=0.95): 
    N = len(S)

    #compute the aggregate spectral features of interest 
    cent = spectral_centroid(S=S,sr=sr,n_fft=N,win_length=N,center=False).item()
    spread = spectral_bandwidth(S=S,sr=sr,n_fft=N,win_length=N,center=False,freq=freqs,p=2).item()/np.sqrt(np.sum(S))
    skew = spectral_bandwidth(S=S,sr=sr,n_fft=N,win_length=N,center=False,freq=freqs,p=3).item()**3/((spread**3)*(np.sum(S)))
    kurt = spectral_bandwidth(S=S,sr=sr,n_fft=N,win_length=N,center=False,freq=freqs,p=4).item()**4/((spread**4)*(np.sum(S)))
    contr = spectral_contrast(S=S,sr=sr,n_fft=N,hop_length=N,center=False,freq=freqs,fmin=freqs[1],n_bands=1,linear=False).flatten()[-1]
    flat = spectral_flatness(S=S,n_fft=N,win_length=N,center=False,power=2.0).item()
    rolloff = spectral_rolloff(S=S,sr=sr,n_fft=N,win_length=N,center=False,roll_percent=roll_percent).item()
    dc = S[0].item()
    maxf = freqs[np.argwhere(S.flatten() == np.max(S[1:]))[-1]][0]
    
    return(pd.Series(data={"centroid":cent,"spread":spread, "skew":skew,"kurtosis":kurt,"contrast":contr,"flatness":flat,"rolloff_point":rolloff,"dc":dc,"maxfreq":maxf}))

def get_spectral_features(freqs, amplitudes): 
    return pd.concat([spectral_feature_compute(amplitudes[i],freqs,sr=1/60,roll_percent=0.95).add_prefix("spectral_").add_suffix(".%s"%(i)) for i in range(len(amplitudes))])

#cepstral features 
def get_cepstrum_features(freqs, amplitudes): 
    return pd.concat([spectral_feature_compute(amplitudes[i],freqs,sr=1/60,roll_percent=0.95).add_prefix("cepstrum_").add_suffix(".%s"%(i)) for i in range(len(amplitudes))])


# =============================================================================
# TRAIN CLASSIFIER
# =============================================================================

def train_classifier(x,y):    
    clf = RandomForestClassifier(max_depth=10, random_state=29, min_samples_leaf =4, max_features = "log2",bootstrap=True, n_jobs = -1, class_weight = {1:flood_weight, 0:noise_weight})
    clf.fit(x,y)
    return clf 

