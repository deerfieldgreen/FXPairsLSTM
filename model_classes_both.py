
from AlgorithmImports import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset

import math
import random
from sklearn.preprocessing import MinMaxScaler    
from sklearn.metrics import confusion_matrix, classification_report



def get_class_distribution(obj):
    count_dict = {
        0: 0,
        1: 0,
        2: 0,
    }
    for i in obj:
        count_dict[i] += 1
    return count_dict


def get_weighted_sampler(y):
    target_list = []
    for t in y:
        target_list.append(t)
    target_list = torch.tensor(target_list)

    class_count = [i for i in get_class_distribution(target_list.cpu().numpy()).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    return (weighted_sampler, class_weights)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True



def get_rnn_dataloader_from_array(X_data, y_data, window_size, batch_size, is_test_loader=False, use_weighted_sampler=False):

    weighted_sampler = None
    class_weights = None
    X, y = [], []
    for i in range(window_size, len(X_data)+1):
        feature = X_data[(i-window_size):i,:]
        target = y_data[i-1]
        X.append(feature)
        y.append(target)
    X = torch.tensor(X).float()
    y = torch.tensor(y).long()
    if is_test_loader:
        data_loader = DataLoader(TensorDataset(X, y), batch_size=1)
    else:
        if use_weighted_sampler:
            # (weighted_sampler, class_weights) = get_weighted_sampler(list(y_data), len(y))
            (weighted_sampler, class_weights) = get_weighted_sampler(y)
            data_loader = DataLoader(TensorDataset(X, y), sampler=weighted_sampler, batch_size=batch_size)
        else:
            data_loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=batch_size)

    return (data_loader, weighted_sampler, class_weights)




def get_torch_rnn_dataloaders(
    col_feature, col_target, train_df, valid_df, test_df, window_size, batch_size, 
    use_weighted_sampler=False, 
    has_test_data=True,
    is_training=True,
    scaler=None,
):

    if is_training:
        X_train = train_df[col_feature].values
        y_train = train_df[col_target].values

        X_val = valid_df[col_feature].values
        y_val = valid_df[col_target].values

    if has_test_data:
        X_test = test_df[col_feature].values
        y_test = test_df[col_target].values

    if is_training:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    if has_test_data:
        X_test = scaler.transform(X_test)

    if is_training:
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)

    if has_test_data:
        X_test, y_test = np.array(X_test), np.array(y_test)

    train_loader = None
    val_loader = None
    weighted_sampler = None
    class_weights = None
    if is_training:
        (train_loader, weighted_sampler, class_weights) = get_rnn_dataloader_from_array(X_train, y_train, window_size, batch_size, is_test_loader=False, use_weighted_sampler=use_weighted_sampler)
        (val_loader, _, _) = get_rnn_dataloader_from_array(X_val, y_val, window_size, batch_size, is_test_loader=False, use_weighted_sampler=False)

    test_loader = None
    if has_test_data:
        (test_loader, _, _) = get_rnn_dataloader_from_array(X_test, y_test, window_size, batch_size, is_test_loader=True)

    return (train_loader, val_loader, test_loader, scaler, weighted_sampler, class_weights)



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
    def __init__(self, input_size, hidden_size, output_size, arc_num=1, use_t2v=True):
        super(GRUmodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
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
            self.layer3 = nn.Linear(256, output_size)

        if self.arc_num == 2:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

        if self.arc_num == 3:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.layer3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

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
    def __init__(self, input_size, hidden_size, output_size, 
                 use_dual_lstm=False, arc_num=0, use_t2v=True):
        super(LSTMmodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
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
            self.layer1 = nn.Linear(hidden_size, output_size)

        if self.arc_num == 1:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, output_size)

        if self.arc_num == 2:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

        if self.arc_num == 3:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.layer3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

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
    epochs, batch_size, learning_rate, window_size, hidden_size, device,
    use_early_stop=False, use_weighted_sampler=False, class_weights=None,
    use_dual_lstm=False, use_gru_model=False,
):

    input_size = len(col_feature)
    output_size = 3
    #use_early_stop = False

    if use_weighted_sampler:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # encoder = RNNEncoder(input_size, hidden_size, device).to(device)
    # decoder = RNNDecoder(hidden_size, output_size).to(device)
    # model = RNNSeq2Seq(encoder, decoder).to(device)

    if use_gru_model:
        model = GRUmodel(input_size, hidden_size, output_size).to(device)
    else:
        model = LSTMmodel(input_size, hidden_size, output_size, use_dual_lstm=use_dual_lstm).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    if use_early_stop:
        early_stopping = EarlyStopping(tolerance=5, min_delta=0.01)

    # print("Begin training.")
    for e in range(1, epochs+1):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            if X_train_batch.shape[0] == 1:
                continue

            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)      
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

        if use_early_stop:
            early_stopping(train_epoch_loss/len(train_loader), val_epoch_loss/len(val_loader))
        # print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

        if use_early_stop and early_stopping.early_stop:
            break

    return model


def get_predictions(test_loader, model, device):
    y_pred_list = []
    y_score_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_score, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_score_list.append(y_pred_score.cpu().numpy())
            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_score_list = [a.squeeze().tolist() for a in y_score_list]
    return (y_pred_list, y_score_list)




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








