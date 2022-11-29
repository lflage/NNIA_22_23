import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

def all_rmse(dataset, n_combinations):
    #comb = dataset.feature_names
    #combination(comb,n_combinations)
    comb_list = combinations([i for i in range(len(dataset.feature_names))],n_combinations)
    rmse_all = {}
    for comb in comb_list:
        # Getting feature names
        selected_features = ""
        for i in comb:
            selected_features += dataset.feature_names[i]+'_'
        # Unpacking indexes
        i1,i2,i3 = comb
        
        features = dataset.data

        # Fiting the selected features
        to_fit = np.stack((features[:,i1], features[:,i2],features[:,i3])).T
        my_LR = LinearRegression().fit(to_fit, housing.target)

        # Feature prediction
        y_hat = my_LR.predict(to_fit)

        # RMSE
        rmse = ((y_hat-dataset.target)**2).mean()
        # Appending Features used and RMSE to a dict
        rmse_all[selected_features] = rmse
    return rmse_all 


def sort_dict(to_be_sorted):
    sorted_dict = sorted(to_be_sorted.items(), key=lambda x:x[1])    
    return dict(sorted_dict)

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

def run_pca(data, pc_num):
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
    out_dict["Selected Features"] = pca.get_feature_names_out
    return out_dict

if __name__=="__main__":

    rmse_dict = all_rmse(housing, 3)
    final_dict = sort_dict(rmse_dict)
    print(final_dict)