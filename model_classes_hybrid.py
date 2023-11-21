#region imports
from AlgorithmImports import *
#endregion
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

import math
import random
import pandas as pd
import numpy as np


def get_rnn_dataloader_from_array(X_data, y_data, window_size, batch_size, is_test_loader=False):
    X, y = [], []
    for i in range(window_size, len(X_data)+1):
        feature = X_data[(i-window_size):i,:]
        target = y_data[i-1]
        X.append(feature)
        y.append(target)
    X = torch.tensor(X).float()
    # y = torch.tensor(y).long()
    y = torch.tensor(y).float()
    
    if is_test_loader:
        data_loader = DataLoader(TensorDataset(X, y), batch_size=1)
    else:
        data_loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=batch_size)

    return data_loader


def get_torch_rnn_dataloaders(
    col_feature, col_target, train_df, valid_df, test_df, window_size, batch_size,
    has_test_data=True,
    is_training=True,
    scaler=None,
):

    if is_training:
        train_data = train_df[[col_target]+col_feature].values

        valid_df_windowed = pd.concat([train_df,valid_df]).copy()
        valid_df_windowed = valid_df_windowed.tail(len(valid_df) + window_size-1)
        valid_data = valid_df_windowed[[col_target]+col_feature].values

    if has_test_data:
        test_data = test_df[[col_target]+col_feature].values

    if is_training:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)

    if has_test_data:
        test_data = scaler.transform(test_data)

    if is_training:
        X_train = train_data[:, 1:]
        y_train = train_data[:, 0]
        X_val = valid_data[:, 1:]
        y_val = valid_data[:, 0]

    if has_test_data:
        X_test = test_data[:, 1:]
        y_test = test_data[:, 0]

    train_loader = None
    val_loader = None
    test_loader = None

    if is_training:
        train_loader = get_rnn_dataloader_from_array(X_train, y_train, window_size, batch_size, is_test_loader=False)
        val_loader = get_rnn_dataloader_from_array(X_val, y_val, window_size, batch_size, is_test_loader=True)

    if has_test_data:
        test_loader = get_rnn_dataloader_from_array(X_test, y_test, window_size, batch_size, is_test_loader=True)

    return (train_loader, val_loader, test_loader, scaler)




def t2v(tau, f, out_features, w, b, w0, b0):
    v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)



class GRUmodel(nn.Module):
    def __init__(self, input_size, hidden_size, arc_num=1, use_t2v=True):
        super(GRUmodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.arc_num = arc_num
        self.use_t2v = use_t2v

        if self.use_t2v:
            self.t2v_layer = SineActivation(in_features=input_size, out_features=16)
            self.layer0 = nn.Linear(16, input_size)

        self.recurrent_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        if self.arc_num == 1:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 1)

        if self.arc_num == 2:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, 1)

        if self.arc_num == 3:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.layer3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
            
        if self.use_t2v:
            x = self.t2v_layer(x)
            x = self.layer0(x)    
             
        o, h = self.recurrent_layer(x)
        h = h.squeeze().unsqueeze(0) if len(h.squeeze().shape) < 2 else h.squeeze()

        if self.arc_num == 1:
            x = self.layer1(h)
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            output = self.layer3(x)


        if self.arc_num in [2,3]:
            x = self.layer1(h)
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            x = self.layer3(x)
            x = self.bn3(x)
            output = self.layer4(x)

        output if len(output.shape) > 1 else output.unsqueeze(0)

        return output




class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size,
                 use_dual_lstm=False, arc_num=1, use_t2v=True):
        super(LSTMmodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_dual_lstm = use_dual_lstm
        self.arc_num = arc_num
        self.use_t2v = use_t2v

        if self.use_t2v:
            self.t2v_layer = SineActivation(in_features=input_size, out_features=16)
            self.layer0 = nn.Linear(16, input_size)

        self.recurrent_layer1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        if self.use_dual_lstm:
            self.recurrent_layer2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        if self.arc_num == 0:
            self.layer1 = nn.Linear(hidden_size, 1)

        if self.arc_num == 1:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 1)

        if self.arc_num == 2:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, 1)

        if self.arc_num == 3:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.layer3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        if self.use_t2v:
            x = self.t2v_layer(x)
            x = self.layer0(x)    

        rx, (hn, cn) = self.recurrent_layer1(x)
        if self.use_dual_lstm:
            rx, (hn, cn) = self.recurrent_layer2(rx)

        if self.arc_num == 0:
            output = self.layer1(rx[:,-1])

        if self.arc_num == 1:
            x = self.layer1(rx[:,-1])
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            output = self.layer3(x)

        if self.arc_num in [2,3]:
            x = self.layer1(rx[:,-1])
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            x = self.layer3(x)
            x = self.bn3(x)
            output = self.layer4(x)

        output if len(output.shape) > 1 else output.unsqueeze(0)
        return output




def get_rnn_model(
    col_feature, train_loader, val_loader,
    epochs, learning_rate, hidden_size, device,
    use_dual_lstm=False, use_gru_model=False,
):

    input_size = len(col_feature)

    if use_gru_model:
        model = GRUmodel(input_size, hidden_size).to(device)
    else:
        model = LSTMmodel(input_size, hidden_size, use_dual_lstm=use_dual_lstm).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_stats = {
        'train': [],
        "val": []
    }

    for e in range(1, epochs+1):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            if X_train_batch.shape[0] == 1:
                continue

            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = F.smooth_l1_loss(y_train_pred, y_train_batch.unsqueeze(1))
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()

        # VALIDATION
        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)      
                val_loss = F.smooth_l1_loss(y_val_pred, y_val_batch.unsqueeze(1))
                val_epoch_loss += val_loss.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))

    return model



def get_predictions(test_loader, model, scaler, col_feature, device):
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = y_test_pred.cpu().squeeze().numpy().item()
            y_pred_list.append(y_test_pred)

    def inverse_transform(y_pred, col_feature):
        extended = np.zeros((len(y_pred), len(col_feature)+1))
        extended[:, 0] = y_pred
        return scaler.inverse_transform(extended)[:, 0]
        
    y_pred = np.array(y_pred_list)
    y_pred = inverse_transform(y_pred, col_feature)
    return y_pred




def get_prediction_hybrid_regression(pred_fundamental, pred_technical, fundamental_mse, technical_mse):
    out = None

    if (pred_technical == 1):
        out = 1
    elif (pred_fundamental == 1):
        out = 1
    elif pred_technical == pred_fundamental:
        out = pred_technical
    else:
        if technical_mse <= fundamental_mse:
            out = pred_technical
        else:
            out = pred_fundamental

    return out



def get_regression_pred_decision(diff, col_target_gains_thres):
    if diff > col_target_gains_thres:
        return 2
    if -diff > col_target_gains_thres:
        return 0
    else:
        return 1


