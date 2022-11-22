# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:06:02 2022

@author: bened
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_reconstr_error(array1,array2):
    """
    For two arrays, computes the MSE, which in this case constitutes the reconstruction error.
    
    array1: original data
    array2: reconstructed data
    
    1. Matrix substraction to get matrix of differences for each datapoint in each variable
    2. Square those differences.
    3. Sum the squared differences along the columns (axis=1) to obtain vector of summed squared differences
    4. Compute mean for vector of summed squared differences = MSE
    
    Return MSE
    """
    return np.sum((array1-array2)**2,axis=1).mean()


def run_pca(data,pc_num):
    """
    For a given dataset and specified number of PCs, run PCA.
    
    data: original dataset
    pc_num: number of principal components (max: number of variables in dataset)
    
    Returns output dictionary of the form:
        
        "Norm data": normalised original data
        "PCA data": dimensionality reduced data
        "Reconstr norm data": the reconstructed data (still normalized)
        "Reconstr data": the reconstructed data (normalization reversed)
        "Reconstr error norm": error between normalized and reconstructed normalized data
        "Reconstr error original": error between original data and reconstructed data (normalization reversed)
        
    """
    
    assert pc_num <= data.shape[1],'Number of principal components higher than number of variables'
    
    out_dict = dict()
    
    scaler = StandardScaler()
    scaler_fitted = scaler.fit(data)
    norm_data =scaler_fitted.transform(data)
    
    pca = PCA(n_components=pc_num)
    pca_fitted = pca.fit(norm_data)
    pca_data = pca_fitted.transform(norm_data)
    
    reconstr_norm_data = pca_fitted.inverse_transform(pca_data) # reconstructed from PCA
    reconstr_data =scaler_fitted.inverse_transform(reconstr_norm_data)
    
    reconstr_error_norm = compute_reconstr_error(norm_data, reconstr_norm_data)
    reconstr_error_original = compute_reconstr_error(data, reconstr_data)
    
    out_dict["Norm data"] = norm_data
    out_dict["PCA data"] = pca_data
    out_dict["Reconstr norm data"] = reconstr_norm_data
    out_dict["Reconstr data"] = reconstr_data
    out_dict["Reconstr error norm"] = reconstr_error_norm
    out_dict["Reconstr error original"] = reconstr_error_original
    
    return out_dict
    