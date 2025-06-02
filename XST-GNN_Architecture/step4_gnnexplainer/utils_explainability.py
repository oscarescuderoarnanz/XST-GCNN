
import torch, numpy as np, pandas as pd
from torch_geometric.data import Data
import torch

def build_data(feats_ft: torch.Tensor | np.ndarray, A_ftft: torch.Tensor, *, device):
    if isinstance(feats_ft, np.ndarray):
        feats_ft = torch.tensor(feats_ft, dtype=torch.float32)
    if feats_ft.ndim == 1:
        T = 14
        F = feats_ft.numel() // T
        feats_ft = feats_ft.view(F, T)
    F, T = feats_ft.shape
    x = feats_ft.reshape(F*T, 1).to(device)
    edge_index = (A_ftft > 0).nonzero(as_tuple=False).t().to(device)
    batch = torch.zeros(F*T, dtype=torch.long, device=device)
    return Data(x=x, edge_index=edge_index, batch=batch)

def load_data(norm, device, split, SG=False, numberOfTimeStep=14):
    """
    Load and preprocess the training, validation, and test data.
    
    Args:
        norm (str): Normalization type.
        device (torch.device): Device to load the data onto (CPU or GPU).
        split (str): Data split name (e.g., 'train', 'val', 'test').
        SG (bool): If True, applies special graph (SG) preprocessing. Default is False.
        numberOfTimeStep (int): Number of time steps to consider. Default is 14.
    
    Returns:
        tuple: Preprocessed tensors for training, validation, and test sets, along with their labels.
    """
    
    # Load raw data
    X_train = np.load("../../DATA/" + split + "/X_train_tensor_" + norm + ".npy")
    X_val = np.load("../../DATA/" + split + "/X_val_tensor_" + norm + ".npy")
    X_test = np.load("../../DATA/" + split + "/X_test_tensor_" + norm + ".npy")

    y_train = pd.read_csv("../../DATA/" + split + "/y_train_" + norm + ".csv")[['individualMRGerm_stac']]
    y_train = y_train.iloc[0:y_train.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_train = torch.tensor(y_train['individualMRGerm_stac'], dtype=torch.float32)

    y_val = pd.read_csv("../../DATA/" + split + "/y_val_tensor_" + norm + ".csv")[['individualMRGerm_stac']]
    y_val = torch.tensor(y_val['individualMRGerm_stac'], dtype=torch.float32)

    y_test = pd.read_csv("../../DATA/" + split + "/y_test_" + norm + ".csv")[['individualMRGerm_stac']]
    y_test = y_test.iloc[0:y_test.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_test = torch.tensor(y_test['individualMRGerm_stac'], dtype=torch.float32)
    
    if SG:
        X_train[X_train == 666] = np.nan
        X_val[X_val == 666] = np.nan
        X_test[X_test == 666] = np.nan

        X_train = np.nanmean(X_train, axis=1)
        X_val = np.nanmean(X_val, axis=1)
        X_test = np.nanmean(X_test, axis=1)

        X_train_vec = torch.tensor(X_train, dtype=torch.float32)
        X_val_vec = torch.tensor(X_val, dtype=torch.float32)
        X_test_vec = torch.tensor(X_test, dtype=torch.float32)
    else:
        X_train[X_train == 666] = 0
        X_val[X_val == 666] = 0
        X_test[X_test == 666] = 0

        # Vectorize each of the train/val/test sets
        n, dim1, dim2 = X_train.shape
        X_train_vec = torch.tensor(X_train.reshape((n, dim1 * dim2)), dtype=torch.float32)

        n, dim1, dim2 = X_val.shape
        X_val_vec = torch.tensor(X_val.reshape((n, dim1 * dim2)), dtype=torch.float32)

        n, dim1, dim2 = X_test.shape
        X_test_vec = torch.tensor(X_test.reshape((n, dim1 * dim2)), dtype=torch.float32)
    
    X_train_vec = X_train_vec.unsqueeze(2)
    X_val_vec = X_val_vec.unsqueeze(2) 
    X_test_vec = X_test_vec.unsqueeze(2) 
    
    if device.type == "cuda":
        return X_train_vec.to(device), X_val_vec.to(device), X_test_vec.to(device), y_train.to(device), y_val.to(device), y_test.to(device)
    else:
        return X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test