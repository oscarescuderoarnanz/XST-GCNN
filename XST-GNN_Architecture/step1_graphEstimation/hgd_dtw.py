import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import random
from joblib import Parallel, delayed

def replace_nan_with_zero(list_of_arrays):
    """
    Replaces NaN values with 0 in each array within the list of arrays.
    
    Parameters:
    list_of_arrays (list): List of NumPy arrays.
    
    Returns:
    list: A list of arrays with NaN values replaced by 0.
    """
    list_without_nan = [np.nan_to_num(arr, nan=0) for arr in list_of_arrays]
    return list_without_nan

def elements_in_list(large_list, small_list):
    """
    Checks if each element in the large list is present in the small list.
    
    Parameters:
    large_list (list): The list to search within.
    small_list (list): The list containing elements to search for.
    
    Returns:
    list: A list of booleans indicating presence of each element.
    """
    return [element in small_list for element in large_list]

def rescale_binary_continuous(df, binary_row, continuous_row):
    """
    Rescales continuous variables based on the binary classification (0 or 1).
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    binary_row (int): The index of the binary row in the DataFrame.
    continuous_row (int): The index of the continuous row in the DataFrame.
    
    Returns:
    DataFrame: The rescaled DataFrame.
    """
    df_aux = df.copy()
    
    max_value = np.max(df.iloc[continuous_row])
    filtered_array = [element for element in df.iloc[continuous_row] if element != -1]
    min_value = np.min(filtered_array)
    
    large_list = df.columns.values

    small_list = np.where(df_aux.iloc[binary_row] == 0)[0]
    df.iloc[binary_row] = df.iloc[binary_row].where(elements_in_list(large_list, small_list), max_value)

    small_list = np.where(df_aux.iloc[binary_row] == 1)[0]
    df.iloc[binary_row] = df.iloc[binary_row].where(elements_in_list(large_list, small_list), min_value)
    
    return df

def rescale_continuous_continuous(df):
    """
    Rescales one continuous variable based on the maximum value of another.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    DataFrame: The rescaled DataFrame.
    """
    max_row = df.values.argmax() // df.shape[1]
    max_value = df.loc[max_row].max()

    if max_row == 0:
        df.loc[1] = list((df.iloc[1] * max_value) / df.loc[1].max())
    else:
        df.loc[0] = list((df.iloc[0] * max_value) / df.loc[0].max())
    
    return df

def binary_binary_hgd(df):
    """
    Implements the Heterogeneous Gower Distance (HGD) for binary-binary variable pairs.
    
    Parameters:
    df (DataFrame): The DataFrame containing the binary variables.
    
    Returns:
    float: The HGD distance for the binary variables.
    """
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0
        
    sj = []
    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] != -1) and (df.iloc[1, i] != -1):
            try:
                sj.append(1 - (np.abs(df.iloc[0,i] - df.iloc[1,i]) / (max_value)))
            except ZeroDivisionError:
                sj.append(1)
            pat_considered += 1

    try:
        output = 1 - (1/pat_considered)*np.sum(sj)
    except ZeroDivisionError:
        output = 0
        
    return output

def binary_continuous_hgd(df):
    """
    Implements the Heterogeneous Gower Distance (HGD) for binary-continuous variable pairs.
    
    Parameters:
    df (DataFrame): The DataFrame containing the binary and continuous variables.
    
    Returns:
    float: The HGD distance for the binary-continuous variables.
    """
    if len(list(df.iloc[0].value_counts().keys())) == 2: 
        df = rescale_binary_continuous(df, 0, 1)
    else:
        df = rescale_binary_continuous(df, 1, 0)

    sj = []
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0
    
    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] != -1) and (df.iloc[1, i] != -1):
            try:
                sj.append(1 - (np.abs(df.iloc[0,i] - df.iloc[1,i]) / (max_value)))
            except ZeroDivisionError:
                sj.append(1)
            pat_considered+=1

    try:
        output = 1- (1/pat_considered)*np.sum(sj)
    except ZeroDivisionError:
        output = 0
        
    return output

def continuous_continuous_hgd(df):
    """
    Implements the Heterogeneous Gower Distance (HGD) for continuous-continuous variable pairs.
    
    Parameters:
    df (DataFrame): The DataFrame containing the continuous variables.
    
    Returns:
    float: The HGD distance for the continuous variables.
    """
    df = rescale_continuous_continuous(df)

    sj = []
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0
        
    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] >= 0) and (df.iloc[1, i] >= 0):
            sj.append(1 - ((np.abs(df.iloc[0,i] - df.iloc[1,i])) / (max_value)))
            pat_considered+=1

    try:
        output = 1- (1/pat_considered)*np.sum(sj)
    except ZeroDivisionError:
        output = 0
        
    return output

def compute_hgd_matrix(X, feature1, feature2, f1, f2, binary, continuous):
    """
    Computes the Heterogeneous Gower Distance (HGD) matrix based on variable types.
    
    Parameters:
    X (ndarray): The input data array.
    feature1 (ndarray): First feature array.
    feature2 (ndarray): Second feature array.
    f1, f2 (str): Feature types (binary or continuous).
    binary (list): List of binary features.
    continuous (list): List of continuous features.
    
    Returns:
    ndarray: HGD matrix.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    hgd_matrix = np.zeros((X.shape[1], X.shape[1]))

    if f1 in binary and f2 in binary:
        for c in range(X.shape[1]):
            for r in range(X.shape[1]):
                df = pd.DataFrame(np.array([feature1[c, :], feature2[r, :]]))
                hgd = binary_binary_hgd(df)
                hgd_matrix[c,r] = hgd

    elif (f1 in binary and f2 in continuous) or (f1 in continuous and f2 in binary):
        for c in range(X.shape[1]):
            for r in range(X.shape[1]):
                df = pd.DataFrame(np.array([feature1[c, :], feature2[r, :]]))
                hgd = binary_continuous_hgd(df)
                hgd_matrix[c,r] = hgd

    elif f1 in continuous and f2 in continuous:
        for c in range(X.shape[1]):
            for r in range(X.shape[1]):
                df = pd.DataFrame(np.array([feature1[c, :], feature2[r, :]]))
                hgd = continuous_continuous_hgd(df)
                hgd_matrix[c,r] = hgd
                
    else:
        raise NameError('Error: Unsupported variable types encountered.')

    return hgd_matrix

def compute_new_dtw(hgd_matrix):
    """
    Computes the Dynamic Time Warping (DTW) distance using the Heterogeneous Gower Distance (HGD) matrix.
    
    Parameters:
    hgd_matrix (ndarray): The HGD matrix.
    
    Returns:
    float: The DTW distance.
    """
    len_ts1 = hgd_matrix.shape[0]
    len_ts2 = hgd_matrix.shape[1]

    cost_matrix = np.full((len_ts1+1, len_ts2+1), np.inf)
    cost_matrix[0, 0] = 0.
    
    for i in range(len_ts1):
        for j in range(len_ts2):
            cost_matrix[i + 1, j + 1] = hgd_matrix[i][j]
            cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                   cost_matrix[i + 1, j],
                                   cost_matrix[i, j])

    return cost_matrix[-1,-1]

def reshape_patients_by_features(data, keys, numberOfTimeStep):
    """
    Reshapes the data based on the number of features and time steps.
    
    Parameters:
    data (ndarray): The original data array.
    keys (list): List of feature names.
    numberOfTimeStep (int): The number of time steps.
    
    Returns:
    ndarray: The reshaped data.
    """
    df = pd.DataFrame(data.reshape(int(data.shape[0]*numberOfTimeStep), data.shape[2]))
    df.replace(666, -1, inplace=True)
    df.columns = keys

    for i in range(len(keys)):
        df_trial = df[keys[i]]
        if i == 0:
            X = np.array(df_trial)
            X = X.reshape((int(df_trial.shape[0]/numberOfTimeStep), numberOfTimeStep)).T
            X = X.reshape(1, numberOfTimeStep, int(df_trial.shape[0]/numberOfTimeStep))

        else:
            X_2 = np.array(df_trial)
            X_2 = X_2.reshape((int(df_trial.shape[0]/numberOfTimeStep), numberOfTimeStep)).T
            X_2 = X_2.reshape(1, numberOfTimeStep, int(df_trial.shape[0]/numberOfTimeStep))

            X = np.append(X, X_2, axis=0)
        
    return X


def diagonal_to_zero(matrix):
    """
    Sets the diagonal elements of a matrix to zero.

    Parameters:
    matrix (ndarray): The input matrix.

    Returns:
    ndarray: The matrix with diagonal elements set to zero.
    """
    return np.array([[0 if i == j else element for j, element in enumerate(row)] for i, row in enumerate(matrix)])


def binary_binary_hgd(df):
    """
    Implements the Heterogeneous Gower Distance (HGD) for binary-binary variable pairs.

    Parameters:
    df (DataFrame): The DataFrame containing the binary variables.

    Returns:
    float: The HGD distance for the binary variables.
    """
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0
        
    sj = []
    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] != -1) and (df.iloc[1, i] != -1):
            try:
                sj.append(1 - (np.abs(df.iloc[0,i] - df.iloc[1,i]) / (max_value)))
            except ZeroDivisionError:
                sj.append(1)
            pat_considered += 1

    try:
        output = 1 - (1/pat_considered) * np.sum(sj)
    except ZeroDivisionError:
        output = 0
        
    return output

def binary_continuous_hgd(df):
    """
    Implements the Heterogeneous Gower Distance (HGD) for binary-continuous variable pairs.

    Parameters:
    df (DataFrame): The DataFrame containing the binary and continuous variables.

    Returns:
    float: The HGD distance for the binary-continuous variables.
    """
    if len(list(df.iloc[0].value_counts().keys())) == 2: 
        df = rescale_binary_continuous(df, 0, 1)
    else:
        df = rescale_binary_continuous(df, 1, 0)

    sj = []
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0
    
    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] != -1) and (df.iloc[1, i] != -1):
            try:
                sj.append(1 - (np.abs(df.iloc[0,i] - df.iloc[1,i]) / (max_value)))
            except ZeroDivisionError:
                sj.append(1)
            pat_considered += 1

    try:
        output = 1 - (1/pat_considered) * np.sum(sj)
    except ZeroDivisionError:
        output = 0
        
    return output

def continuous_continuous_hgd(df):
    """
    Rescales and implements the Heterogeneous Gower Distance (HGD) for continuous-continuous variable pairs.

    Parameters:
    df (DataFrame): The DataFrame containing the continuous variables.

    Returns:
    float: The HGD distance for the continuous variables.
    """
    df = rescale_continuous_continuous(df)

    sj = []
    max_value = np.max(np.abs(df[df != -1].iloc[0] - df[df != -1].iloc[1]))
    if np.isnan(max_value):
        max_value = 0
        
    pat_considered = 0
    for i in range(df.shape[1]):
        if (df.iloc[0, i] >= 0) and (df.iloc[1, i] >= 0):
            sj.append(1 - ((np.abs(df.iloc[0,i] - df.iloc[1,i])) / (max_value)))
            pat_considered += 1

    try:
        output = 1 - (1/pat_considered) * np.sum(sj)
    except ZeroDivisionError:
        output = 0
        
    return output

def hgd_distance(feature1, feature2, f1, f2, binary, continuous):
    """
    Computes the Heterogeneous Gower Distance (HGD) between two features based on their types (binary or continuous).

    Parameters:
    feature1 (ndarray): First feature array.
    feature2 (ndarray): Second feature array.
    f1, f2 (str): Feature types (binary or continuous).
    binary (list): List of binary features.
    continuous (list): List of continuous features.

    Returns:
    float: The HGD distance between the two features.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    if f1 in binary and f2 in binary:
        df = pd.DataFrame(np.array([feature1, feature2]))
        hgd_distance = binary_binary_hgd(df)

    elif (f1 in binary and f2 in continuous) or (f1 in continuous and f2 in binary):
        df = pd.DataFrame(np.array([feature1, feature2]))
        hgd_distance = binary_continuous_hgd(df)

    elif f1 in continuous and f2 in continuous:
        df = pd.DataFrame(np.array([feature1, feature2]))
        hgd_distance = continuous_continuous_hgd(df)
                
    else:
        raise NameError('Hi, we have a problem! Stop!')

    return hgd_distance
