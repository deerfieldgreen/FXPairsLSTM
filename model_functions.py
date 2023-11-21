
from AlgorithmImports import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset

import math
import random
from sklearn.preprocessing import MinMaxScaler    
from sklearn.metrics import confusion_matrix, classification_report


def set_seed(seed: int = 100) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


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











