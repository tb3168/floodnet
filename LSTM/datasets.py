#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:14:56 2025

@author: tanvibansal
"""
import os
import pandas as pd
import numpy as np
import torch
import random
import numpy as np
import time
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class eventDataset(Dataset):
    """
    pytorch style dataloader
    """
    def __init__(self, data, k, transform=None):
        self.data = data.loc[data.k == k].dropna(subset = "signal_k_partial")
        self.transform = transform
        self.k = k
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_idx = self.data.iloc[idx]
        signal = data_idx["signal_k_partial"]["depth_raw_mm"]
        signal = torch.tensor(signal,dtype=torch.float32)
        label = torch.tensor(int(data_idx["class"] == "flood"))
        uuid = data_idx.name
        return signal, label, uuid

def collate_batch(batch, pad_idx=0):
    signals, labels, uuid = zip(*batch)
    padded_sequences = pad_sequence(signals, batch_first=True, padding_value=pad_idx)
    labels = torch.stack(labels)
    return padded_sequences, labels, uuid

#define function to apply tokenization and get train and test dataloaders
def get_dataloaders(ds, k, batch_size, num_workers, shuffle=True):
    dataset = eventDataset(ds, k)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,collate_fn=lambda batch: collate_batch(batch))
    return dataloader

def get_sequences_and_labels(ds,k):
    ds_k = ds.loc[ds.k == k].dropna()
    sequences = ds_k.signal_k_partial.apply(lambda x: torch.tensor(x["depth_raw_mm"]))
    labels = (ds_k["class"] == "flood").astype("int").values

    max_len = sequences.apply(len).max()
    padded_arrays = sequences.apply(lambda x: np.pad(x, (0, max_len - len(x)), mode='constant'))
    tensor_list = [torch.tensor(arr, dtype=torch.float32) for arr in padded_arrays]
    sequences_tensor = torch.stack(tensor_list)
    
    return sequences_tensor, labels

def get_full_sequences_and_labels(ds):
    #ds_k = ds.loc[ds.k == k].dropna()
    sequences = ds.signal.apply(lambda x: torch.tensor(x["depth_raw_mm"]))
    labels = (ds["class"] == "flood").astype("int").values

    max_len = sequences.apply(len).max()
    padded_arrays = sequences.apply(lambda x: np.pad(x, (0, max_len - len(x)), mode='constant'))
    tensor_list = [torch.tensor(arr, dtype=torch.float32) for arr in padded_arrays]
    sequences_tensor = torch.stack(tensor_list)
    
    return sequences_tensor, labels

def extract_event_timeseries(label_df, buffer):
    """
    helper function for the pytorch dataloader below

    """
    event_ts = []
    sensor_df_fp_parent = "/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/data-1015"
    sensor_df_fps = os.listdir(sensor_df_fp_parent)
    for i in sensor_df_fps:
        label_df_sensor_i = label_df.loc[label_df.deployment_id == i].sort_values(by="start_time")
        if len(label_df_sensor_i) > 0:
            #read in all the data .csvs for the sensor of interest and set the time as index
            sensor_df_fp = sensor_df_fp_parent + "/%s"%(i)
            sensor_df_names = pd.Series([sensor_df_fp + "/" + n for n in os.listdir(sensor_df_fp)])
            sensor_df = pd.concat(list(sensor_df_names.apply(lambda x: pd.read_csv(x))))
            sensor_df["time"] = pd.to_datetime(sensor_df["time"],format="ISO8601")#"%Y-%m-%d %H:%M:%S.%f+%z")
            sensor_df.set_index("time",inplace=True)
            sensor_df.sort_index(inplace=True)
            
            #start parsing each event
            def extract_event_measurements(x,sensor_df, buffer):
                #extract event timeseries + historical buffer
                x0 = (sensor_df.index > x.start_time).nonzero()[0].min() #index location of first measurement in event
                if x0 - buffer < 0:
                    start_time_buffer = sensor_df.index[0]
                else:
                    start_time_buffer = sensor_df.index[x0 - buffer]
                x_df = sensor_df.loc[(sensor_df.index >= start_time_buffer) & (sensor_df.index <= x.end_time),"depth_raw_mm"]
                
                #drop nans
                inds_to_drop = np.isnan(x_df)
                if sum(inds_to_drop) > 0:
                    x_df = x_df[~inds_to_drop]
                #make timeseries dict if we have the event data
                if len(x_df) > 0:
                    x_input = {"time":((x_df.index.values - x_df.index.values[0])/1000000000).astype("int64"),"depth_raw_mm":x_df.values}
                #dont make output if event max depth is less than 30
                elif x_df.max() <= 30:
                    x_input = None
                else:
                    x_input = None
                return x_input
            
            event_ts_i = label_df_sensor_i.apply(lambda x: extract_event_measurements(x,sensor_df,buffer),axis=1)
            event_ts.append(event_ts_i)
        print("%s complete"%(i))
    return event_ts


class SignalAugmentor:
    """
    data augmentation class for upsampling flood events
    """
    def __init__(self, jitter_std=0.02, scale_range=(0.8, 1.2),
                 shift_max=0.1, permute_segments=4,
                 magnitude_warp_k=4, magnitude_std=0.2,
                 time_warp_k=4, time_warp_std=0.2,
                 apply_prob=0.5,stretch_range=(0.8, 1.2)):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.permute_segments = permute_segments
        self.magnitude_warp_k = magnitude_warp_k
        self.magnitude_std = magnitude_std
        self.time_warp_k = time_warp_k
        self.time_warp_std = time_warp_std
        self.apply_prob = apply_prob
        self.stretch_range = stretch_range

    def jitter(self, x):
        return x + torch.randn_like(x) * self.jitter_std

    def scaling(self, x):
        factor = random.uniform(*self.scale_range)
        return x * factor

    def time_shift(self, x):
        max_shift = int(self.shift_max * x.size(-1))
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(x, shifts=shift, dims=-1)

    def permutation(self, x):
        print(torch.chunk(x, self.permute_segments, dim=-1))
        segs = list(torch.chunk(x, self.permute_segments, dim=-1))
        random.shuffle(segs)
        return torch.cat(segs, dim=-1)

    def magnitude_warp(self, x):
        """Smooth amplitude modulation using cubic spline"""
        time = np.arange(x.shape[-1])
        warp = self._generate_random_curve(len(time), self.magnitude_warp_k, self.magnitude_std)
        return x * torch.tensor(warp, dtype=x.dtype, device=x.device)

    def time_warp(self, x):
        """Smooth non-linear time distortion using cubic spline"""
        orig_steps = np.arange(x.shape[-1])
        random_curve = self._generate_random_curve(len(orig_steps), self.time_warp_k, self.time_warp_std)
        distorted_steps = np.clip(np.cumsum(random_curve), 0, x.shape[-1]-1)
        distorted_steps = (distorted_steps - distorted_steps.min()) / (distorted_steps.max() - distorted_steps.min()) * (x.shape[-1] - 1)

        x_np = x.squeeze().cpu().numpy()
        f = CubicSpline(np.arange(len(x_np)), x_np)
        warped = f(distorted_steps)
        return torch.tensor(warped, dtype=x.dtype, device=x.device).unsqueeze(0)
    
    def time_stretch(self, x):
            stretch_factor = random.uniform(*self.stretch_range)
            original_length = x.shape[1]
            new_length = int(original_length * stretch_factor)
            #print(x.shape, original_length, stretch_factor)
            x_stretched = F.interpolate(x.unsqueeze(0), size=new_length, mode='linear', align_corners=False)
            x_stretched = x_stretched.squeeze()
    
            # Resize back to original length if needed (optional)
            #if new_length != original_length:
            #    x_stretched = F.interpolate(x_stretched.unsqueeze(0), size=original_length, mode='linear', align_corners=False).squeeze()
            return x_stretched

    def _generate_random_curve(self, length, knots, std):
        """Helper to generate smooth random warping curves"""
        orig_x = np.linspace(0, length, num=knots)
        rand_y = np.random.normal(loc=1.0, scale=std, size=knots)
        interp = CubicSpline(orig_x, rand_y)
        return interp(np.arange(length))

    def __call__(self, x):
        """ x: (C, T) tensor or (T,) if 1D """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if random.random() < self.apply_prob:
            x = self.jitter(x)
        if random.random() < self.apply_prob:
            x = self.scaling(x)
        if random.random() < self.apply_prob:
            x = self.time_shift(x)
        #if random.random() < self.apply_prob:
        #    x = self.permutation(x)
        if random.random() < self.apply_prob:
            x = self.magnitude_warp(x)
        if random.random() < self.apply_prob:
            x = self.time_warp(x)
        if random.random() < self.apply_prob:
            x = self.time_stretch(x)

        return x


class dataLoader:
    """
    pytorch data loader for our dataset
    """
    def __init__(self,ev_df, buffer):
        self.ev_df = ev_df
        self.buffer = buffer 
        
    def load_data(self):
        event_ts = extract_event_timeseries(self.ev_df, self.buffer)
        event_df = self.ev_df.copy(deep=True).drop(columns=["annotated_by","annotation_session","date_created","date_updated","event_confidence","id","dev_id","last_depth_proc_mm","max_depth_proc_mm","mean_depth_proc_mm","msg_count","sum_precip_5min_mm","tidally_influenced"])
        event_df["class"] = event_df.label.apply(lambda x: "flood" if x == "flood" else "noise")
        event_signal_df = pd.concat(event_ts).to_frame().rename(columns={0:"signal"})
        event_df = event_df.join(event_signal_df).set_index("uuid")
        self.event_df = event_df
    
    def augment_floods(self,multiplier):
        augmentor = SignalAugmentor()
        df = self.event_df.copy(deep=True)
        flood_ds = df.copy(deep=True).loc[df.label=="flood"]

        def create_augmentation_df(x,multiplier):
            x = x.copy(deep=True)
            signal_time = x.signal["time"]
            signal_tensor = torch.tensor(x.signal["depth_raw_mm"],dtype=torch.float32)
            augmented_signal = []
            for i in range(multiplier):
                augmented_s = augmentor(signal_tensor).numpy()
                adj_length = len(augmented_s) - len(signal_tensor)
                if adj_length > 0:
                    augmented_t = np.arange(1,adj_length)*60 + signal_time[-1]
                elif adj_length < 0:
                    augmented_t = signal_time[:adj_length]
                else:
                    augmented_t = signal_time
                augmented_signal.append({"time": augmented_t, "depth_raw_mm":augmented_s})
            x = x.to_frame().T
            x.signal = [augmented_signal]
            x = x.explode("signal")
            
            #create new uuids w/ length >7 to represent augmented floods - append number onto back of original event uuid for backtracing 
            uuids = np.concatenate([np.unique(x.reset_index()["index"].values)*10 + np.arange(10),
            np.unique(x.reset_index()["index"].values)*100 + np.arange(10,multiplier)])
            
            x.set_index(uuids,inplace=True)
            return x

        augmented_df = [create_augmentation_df(flood_ds.iloc[i],multiplier) for i in range(len(flood_ds))]
        augmented_df = pd.concat(augmented_df)

        df_aug = pd.concat([df,augmented_df])
        self.event_df = df_aug
        
    def k_partial_window(self, x, buffer):
        if x.n >= x.k + buffer:
            return {"time":x.signal["time"][0:x.k + buffer], "depth_raw_mm":x.signal["depth_raw_mm"][0:x.k + buffer].astype("float")}

    def k_partial_data(self,k_):    
        train_exp = self.event_df.copy(deep=True)
        train_exp["n"] = self.event_df.apply(lambda x: len(x.signal["time"]),axis=1)
        train_exp["k"] = [k_]*len(train_exp)
        train_exp = train_exp.explode("k")
        train_signal = train_exp.loc[:,["deployment_id","signal","label","class","n","k"]]
        train_signal["signal_k_partial"] = train_signal.apply(lambda x: self.k_partial_window(x,self.buffer),axis=1)
        #self.train_signal = train_signal
        return train_signal
