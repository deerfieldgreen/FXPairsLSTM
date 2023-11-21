import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset

import os
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report



def get_upper_threshold(close):
    difference = close.diff()
    difference[0] = 0
    difference = difference.abs()
    bins = pd.cut(difference, bins=10)
    bins = bins.value_counts().to_frame().reset_index()
    bins["index"] = bins["index"].apply(lambda x: x.right)
    bins = bins.to_numpy()
    percentile_count = len(difference) * 0.85
    count = 0
    for i in range(10):
        count += bins[i, 1]
        if count > percentile_count:
            return bins[i, 0]


def get_entropy(labels, base=None):
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    base = math.e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()


def get_threshold(close):
    difference = close.diff()
    difference = difference.drop(0)
    difference = difference.tolist()
    close.name = "index"
    threshold = 0
    thres_upper_bound = get_upper_threshold(close)
    temp_thres = 0
    best_entropy = -float('inf')

    while temp_thres < thres_upper_bound:
        labels = []
        for diff in difference:
            if diff > temp_thres:
                labels.append(2)
            elif -diff > temp_thres:
                labels.append(1)
            else:
                labels.append(0)
        entropy = get_entropy(labels)
        if entropy > best_entropy:
            best_entropy = entropy
            threshold = temp_thres
        temp_thres = temp_thres + 0.00001
    return np.round(threshold,5)





def get_prediction_hybrid(row, original=False):
    out = None

    if original:
        if (row['pred_technical'] == 1):
            out = 1
        elif (row['pred_fundamental'] == 1):
            out = 1
        elif row['pred_technical'] == row['pred_fundamental']:
            out = row['pred_technical']
        else:
            if row['score_technical'] >= row['score_fundamental']:
                out = row['pred_technical']
            else:
                out = row['pred_fundamental']
    else:
        if row['pred_technical'] == row['pred_fundamental']:
            out = row['pred_technical']
        else:
            out = 1

    return out


def get_prediction_hybrid_max(row):
    out = None
    if row['score_technical'] >= row['score_fundamental']:
        out = row['pred_technical']
    else:
        out = row['pred_fundamental']

    return out


def get_prediction_hybrid_greedy(row):
    out = None
    if row['pred_technical'] == row['pred_fundamental']:
        out = row['pred_technical']
    elif row['pred_technical'] == 1:
        out = row['pred_fundamental']
    elif row['pred_fundamental'] == 1:
        out = row['pred_technical']
    else:
        if row['score_technical'] >= row['score_fundamental']:
            out = row['pred_technical']
        else:
            out = row['pred_fundamental']

    return out




def get_prediction_hybrid_regression(row, technical_mse, fundamental_mse):
    out = None

    if (row['pred_technical'] == 1):
        out = 1
    elif (row['pred_fundamental'] == 1):
        out = 1
    elif row['pred_technical'] == row['pred_fundamental']:
        out = row['pred_technical']
    else:
        if technical_mse <= fundamental_mse:
            out = row['pred_technical']
        else:
            out = row['pred_fundamental']

    return out



def get_profit_accuracy(test_df, col_pred, col_target_gains):

    true_dec = np.sum((test_df[col_pred] == 0) * (test_df[col_target_gains] < 0) * 1)
    true_inc = np.sum((test_df[col_pred] == 2) * (test_df[col_target_gains] > 0) * 1)
    false_dec = np.sum((test_df[col_pred] == 0) * (test_df[col_target_gains] > 0) * 1)
    false_inc = np.sum((test_df[col_pred] == 2) * (test_df[col_target_gains] < 0) * 1)

    profit_accuracy = (true_dec + true_inc) / (true_dec + true_inc + false_dec + false_inc)
    pred_count = true_dec + true_inc
    total_count = len(test_df)

    short_accuracy = true_dec / (true_dec + false_dec)
    long_accuracy = true_inc / (true_inc + false_inc)

    return (profit_accuracy, pred_count, total_count, short_accuracy, long_accuracy)




def get_regression_pred_decision(diff, col_target_gains_thres):
    if diff > col_target_gains_thres:
        return 2
    if -diff > col_target_gains_thres:
        return 0
    else:
        return 1
