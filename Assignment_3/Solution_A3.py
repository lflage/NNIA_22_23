# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:06:02 2022

@author: bened
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import fetch_california_housing
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


housing = fetch_california_housing()

# Converting sklearn Bunch object into pandas data frame
# Without target column
housing_df = pd.DataFrame(data=housing['data'], columns=housing['feature_names'])


# There have to be 56 unique combinations for taking subsets of 3 (k) from 8 (n) features:
# n!/k!(n-k)! = 8!/3!5! = 56
def get_subsets(features):
    comb_all= list(itertools.combinations(features,3)) # all combinations as tuples
    comb_all = [list(t) for t in comb_all] # format for indexing in pd data frame
    return comb_all

housing_subsets = get_subsets(housing_df.columns)
house_target = housing['target'] # type np array


def lin_reg(data,target):
    feature_names = list(data.columns)
    X = data.to_numpy() # shape: (20640,3)
    y = target # shape: (20640,)
    lin_regressor = LinearRegression()
    lin_regressor.fit(X,y)
    y_hat = lin_regressor.predict(X)
    mse = mean_squared_error(y, y_hat)
    
    return ((feature_names,mse))


def lin_reg_subsets():
    min_error = 1000
    for subs in housing_subsets:
        l_r= lin_reg(housing_df[subs],house_target)
        print(l_r)
        if l_r[1] < min_error:
            min_error = l_r[1]
            min_error_set = l_r[0]
    print('\n')
    print('Min error set: ',min_error_set,'\n Min error: ',min_error)
    
    return
    



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
    out_dict["Component corrs"] = pca.components_ # Correlation between PCs and original features 
    
    # Selecting the strongest feature correlation for each PC
    out_dict["Strongest corrs"] = []
    for v in out_dict["Component corrs"]:
        v_abs = np.absolute(v)
        v_max= np.max(v_abs)
        ind = np.where(v==v_max)[0][0] # returns int
        out_dict["Strongest corrs"].append(ind)
    
    
    return out_dict


def house_pca_reg():
    
    house_pca = run_pca(housing_df, 3)
    
    # Setting names for the PCs by selecting the strongest feature correlation
    pca_names = []
    for i,k in enumerate(house_pca["Strongest corrs"]):
        pc_num = i+1
        feat_name = housing_df.columns[k]
        pca_names.append('PC '+str(pc_num)+' '+feat_name)
    
    pca_frame = pd.DataFrame(data=house_pca["PCA data"],columns=pca_names)
    print(lin_reg(pca_frame,house_target))
    