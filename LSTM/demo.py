#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:28:19 2025

@author: tanvibansal
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from LSTM.training import training_loop, testing_loop, calculate_accuracy, calculate_precision_recall
from LSTM.results import plot_confusion_matrix_with_metrics, predictions_to_df
from LSTM.datasets import *
from LSTM.models import chronosForecast, LSTMClassifier
root_fp = "data/"
label_df = pd.read_csv(root_fp + "sensor_events_uuid_tidy.csv")
train,test,val = [pd.read_csv(f"{root_fp}{i}.csv") for i in ["train","test","val"]]

 #%% load parent data sets and super parameters 
buffer = 0 
k_vals = list(np.arange(1,32,2))
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
#%% run training loop with hyperparameters
#hyper parameters 
k = 18
model_size = "t5-small"
embed_dims = {"t5-large":1024, "t5-small":512, "t5-base": 768}
embedding_dim = embed_dims[model_size]
hidden_size = 16
num_classes = 1
num_epochs = 20
dropout = .076
num_layers = 5
batch_size = 8
lr = .0001
pos_weight = 3.0 

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

#get dataloaders
train_dataloader, val_dataloader, test_dataloader = [get_dataloaders(d, k, batch_size, num_workers=0, shuffle=True) for d in [train_ds, val_ds, test_ds]]

#define model, optimizer, scheduler
forecast = chronosForecast(device = device, model=model_size)
model = LSTMClassifier(embedding_dim, hidden_size, num_classes, num_layers, device, model_size, dropout = dropout).to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=lr, capturable=False)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=num_epochs,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.3
)

#run training/eval loop and store results
train_losses = []
train_accs = []
train_labels = [] 
train_preds = [] 
train_probs = []
train_precisions = []
train_recalls = []

val_losses = []
val_accs = []
val_labels = [] 
val_preds = [] 
val_probs = []
val_precisions = []
val_recalls = []

for epoch in range(num_epochs):
    train_loss, train_acc, train_prec, train_rec, train_label, train_pred, train_prob = training_loop(model, "train", train_dataloader, optimizer, criterion, device, scheduler)
    val_loss, val_acc, val_prec, val_rec, val_label, val_pred, val_prob = training_loop(model, "val", val_dataloader, optimizer, criterion, device, scheduler)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_labels.append(train_label)
    train_preds.append(train_pred)
    train_probs.append(train_prob)
    train_precisions.append(train_prec)
    train_recalls.append(train_rec)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_labels.append(val_label)
    val_preds.append(val_pred)
    val_probs.append(val_prob)
    val_precisions.append(val_prec)
    val_recalls.append(val_rec)

        
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:.4f}, Val. Acc: {val_acc:.2f}%, Val. Precision: {val_prec:.2f}%, Val. Recall: {val_rec:.2f}%')

    if epoch > 4:
        val_loss_arr = np.array(val_losses)
        val_loss_pct_change = np.diff(val_loss_arr)*100/val_loss_arr[:-1] 
        if (np.abs(val_loss_pct_change[-3:]) >= 0.1).sum() == 0: # none of the last 3 epochs have validation loss gt 0.1%
            print(f"Early stopping at epoch: {epoch+1:02}")
            break

#%% plot training loss and test results
fig,ax = plt.subplots(ncols=2,figsize=(10,4))
ax[0].plot(range(1, len(train_losses)+1), train_losses, label="train")
ax[0].plot(range(1, len(val_losses)+1), val_losses, label="val")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].grid()
ax[0].set_title("Train/Test Loss vs Epochs for Single-Layer LSTM")
ax[0].legend()

fpr, tpr, _ = roc_curve(val_labels[-1].numpy(), val_probs[-1].numpy())
accuracy = val_accs[-1]
auroc = auc(fpr, tpr)

# Plot ROC Curve
ax[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve:  AUC = {auroc:.2f}, ACC = {accuracy:.2f}')
ax[1].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('ROC Curve for Single-Layer LSTM')
ax[1].legend(loc='lower right')
ax[1].grid()

class_mapping = {
    0: "Noise",    # Wake
    1: "Flood"
}

test_loss, test_acc, test_prec, test_rec, test_label, test_pred, test_prob, test_uuids = testing_loop(model, test_dataloader, criterion, device)
plot_confusion_matrix_with_metrics(test_label, test_pred, test_prec, test_rec, class_mapping)
print(f"precision: {test_prec:.2f}, recall: {test_rec:.2f}")

#%% run failure analysis
from LSTM.failure_analysis import get_false_negative_df
false_neg_ids = test_uuids[np.argwhere((test_label != test_pred) & (test_label == 1)).numpy().flatten()].numpy()
fn_df = get_false_negative_df(false_neg_ids, test_ds)
#%%% evaluation
from LSTM.evaluation import events_result_table, get_performance_metrics, evaluate_results
label_mappings = {
    "blip":"blip",
    "flood":"flood",
    "noise":"noise",
    "box":"noise",
    "misc":"noise",
    "pulse-chain":"noise"
    }
event_results_table, recall, precision, specificity = evaluate_results(test_ds, test_prob, test_pred, test_uuids, label_mappings = label_mappings)
event_results_table
recall, precision, specificity


