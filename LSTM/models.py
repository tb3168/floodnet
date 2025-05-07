#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:09:45 2025

@author: tanvibansal
"""
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd 
from chronos import BaseChronosPipeline, ChronosBoltPipeline, ChronosPipeline


#define forecasting model class
class chronosForecast:
    def __init__(self,device = "cpu", model="t5-base"):
        self.pipeline = BaseChronosPipeline.from_pretrained(
        f"amazon/chronos-{model}",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
        device_map=device,  # use "cpu" for CPU inference
        torch_dtype=torch.float32,
        )
    def predict_quantiles(self, df, length=64):
        quantiles, mean = self.pipeline.predict_quantiles(
            context=torch.tensor(df),
            prediction_length=length,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        forecast_index = range(len(df), len(df) + 64)
        low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]
        return quantiles, mean, forecast_index, low, median, high
    def embed(self, df):
        embeddings, tokenizer_state = self.pipeline.embed(df)
        return embeddings,tokenizer_state

class LSTMClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes, num_layers, device, forecast_model_size, dropout = 0.2):
        super(LSTMClassifier, self).__init__()
        self.forecast = chronosForecast(device = device, model=forecast_model_size)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, num_layers = num_layers, dropout = dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.num_layers = num_layers
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        embedding, _ = self.forecast.embed(x)
        embedding = embedding.to(self.device)
        #print(embedding.shape)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
       # print(h0.shape, c0.shape)
        x, (h_n, c_n) = self.lstm(embedding.to(self.device), (h0, c0))
        x_h = h_n[-1]#torch.mean(x, dim=1)#  # final hidden layer used for classification       
        out = self.fc(x_h)
        return out