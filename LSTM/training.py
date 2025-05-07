#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:46:33 2025

@author: tanvibansal
"""
import time
import torch
import numpy as np 

def training_loop(model, phase, dataloader, optimizer, criterion, device, scheduler):
    start_time = time.time()

    #initialize holders for batch results
    epoch_loss = []
    logits = []
    classes = []
    predictions = []

    #put the model in train or eval mode
    if phase == "train":
        model.train()
    else:
        model.eval()

    #training/val operations
    for signals, label, uuid in dataloader:   
        #initialize variables
        optimizer.zero_grad()
        signals = signals
        labels = label.to(device)
        #get model output
        outputs = model(signals)

        #get loss and append relevant outputs to lists
        loss = criterion(outputs.squeeze(1), labels.float())
        batch_logits = outputs.cpu().detach()
        batch_labels = labels.cpu().detach()
        batch_predictions = (torch.sigmoid(outputs).cpu().detach() >= 0.5).to(torch.int32)

        epoch_loss.append(loss.item())
        logits.append(batch_logits)
        classes.append(batch_labels)
        predictions.append(batch_predictions)

        #backpropagate if training
        if phase == "train":
            loss.backward()
            optimizer.step()
        #step scheduler if validating
        else:
            if scheduler:
                scheduler.step()
   
    #shape epoch outputs
    ypred = torch.cat(predictions).squeeze(1)
    ytrue = torch.cat(classes)
    yhat = torch.cat(logits)

    #compute epoch metrics
    loss = np.mean(epoch_loss)
    acc = calculate_accuracy(ypred, ytrue)
    precision, recall = calculate_precision_recall(ypred, ytrue)
    epoch_time = time.time() - start_time 
    
    print(f"{phase} time: {epoch_time:.2f}, {phase} loss: {loss:.2f}, {phase} accuracy: {acc:.2f}")
    return loss, acc, precision, recall, ytrue, ypred, yhat

def testing_loop(model, dataloader, criterion, device):
    start_time = time.time()

    #initialize holders for batch results
    epoch_loss = []
    logits = []
    classes = []
    predictions = []
    uuids = []

    model.eval()

    #training/val operations
    for signals, label, uuid in dataloader:   
        #initialize variables
        signals = signals
        labels = label.to(device)
        #get model output
        outputs = model(signals)

        #get loss and append relevant outputs to lists
        loss = criterion(outputs.squeeze(1), labels.float())
        batch_logits = outputs.cpu().detach()
        batch_labels = labels.cpu().detach()
        batch_predictions = (torch.sigmoid(outputs).cpu().detach() >= 0.5).to(torch.int32)
        batch_uuids = torch.tensor(uuid)

        epoch_loss.append(loss.item())
        logits.append(batch_logits)
        classes.append(batch_labels)
        predictions.append(batch_predictions)
        uuids.append(batch_uuids)
   
    #shape epoch outputs
    ypred = torch.cat(predictions).squeeze(1)
    ytrue = torch.cat(classes)
    yhat = torch.cat(logits)
    yid = torch.cat(uuids)

    #compute epoch metrics
    loss = np.mean(epoch_loss)
    acc = calculate_accuracy(ypred, ytrue)
    precision, recall = calculate_precision_recall(ypred, ytrue)
    epoch_time = time.time() - start_time 
    
    print(f"test time: {epoch_time:.2f}, test loss: {loss:.2f}, test accuracy: {acc:.2f}")
    return loss, acc, precision, recall, ytrue, ypred, yhat, yid

def calculate_accuracy(predictions, labels):
    correct_predictions = (predictions.numpy() == labels.numpy()).sum()
    #print(predictions, labels)
    accuracy = correct_predictions / len(labels) 
    return accuracy

def calculate_precision_recall(predictions, labels):
    tp = np.argwhere((predictions.numpy() == labels.numpy()) & (labels.numpy() ==1)).flatten().shape[0]
    tn = np.argwhere((predictions.numpy() == labels.numpy()) & (labels.numpy() !=1)).flatten().shape[0]
    fn = np.argwhere((predictions.numpy() != labels.numpy()) & (labels.numpy() == 1)).flatten().shape[0]
    fp = np.argwhere((predictions.numpy() != labels.numpy()) & (labels.numpy() == 0)).flatten().shape[0]
    
    if tp + fp == 0:
        precision = np.nan
    else:
        precision = tp/(tp+fp)
    if tp + fn == 0:
        recall = np.nan
    else:
        recall = tp/(tp+fn)
    return precision, recall