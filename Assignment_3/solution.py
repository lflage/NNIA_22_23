import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

def all_rmse(dataset, n_combinations):
    #comb = dataset.feature_names
    #combination(comb,n_combinations)
    comb_list = combinations([i for i in range(len(dataset.feature_names))],n_combinations)
    final_dict = {}
    for comb in comb_list:
        # Getting feature names
        selected_features = ""
        for i in comb:
            selected_features += dataset.feature_names[i]+'_'
        # Unpacking indexes
        i1,i2,i3 = comb
        
        features = dataset.data

        print(features[:,i1].ndim)
        # Fiting the selected features
        to_fit = np.concatenate((features[:,i1], features[:,i2],features[:,i3]))

        print(to_fit.shape)
        print(to_fit)
        my_LR = LinearRegression().fit(to_fit, housing.target)
        y_hat = my_LR.predict(dataset.data)
        rmse = np.sqrt(((y_hat-dataset.target)**2).mean())

        final_dict[selected_features] = rmse
    return final_dict



combs = all_rmse(housing, 3)
print(combs)

