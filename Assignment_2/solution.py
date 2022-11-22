import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.datasets import fetch_california_housing


def solver1():
    # Reading csv file into a pandas DataFrame
    birds = pd.read_csv('./birds.csv')

    # # Initializing the Standardizer
    # scaler = StandardScaler()

    # # Fitting the data to the stantardizer
    # scaler.fit(birds)

    # # Stardizing
    # scaled_birds = scaler.transform(birds)

    # Initializing PCA

    my_pca = PCA(n_components=1)
    # my_pca = my_pca.fit(birds)

    reduced_my_pca = my_pca.fit_transform(birds)
    reconstruct_birds = my_pca.inverse_transform(reduced_my_pca)
    
    mse = mean_squared_error(reconstruct_birds, birds)

    return("MSE: {}".format(mse))


def plotter():
    # - Original data
    birds = pd.read_csv('./birds.csv')
    plt.scatter(birds['BodyMass'], birds['Wingspan'])
    plt.ylabel('Wingspan')
    plt.xlabel('BodyMass')
    plt.xlabel("Boddy Mass")
    plt.ylabel("Wing span")
    plt.show()

    # - Pre-processed data

    scaler = StandardScaler()
    standardized_birds = scaler.fit_transform(birds)
    # print(type(scaler))
    scalerx = [x[0] for x in standardized_birds]
    scalery = [x[1] for x in standardized_birds]
    
    plt.scatter(scalerx,scalery)
    plt.show()

    # - Data projected into 1D using PCA
    my_pca = PCA(n_components=1)
    pca_birds = my_pca.fit_transform(birds)
    # assert(len(pca_birds)==len(my_pca))
    plt.scatter(pca_birds, y=np.zeros(len(pca_birds)))
    plt.show()
    
    # - Reconstructed data
    reconstruct_scalar = my_pca.inverse_transform(pca_birds)
    reconstruct_birds = scaler.inverse_transform(reconstruct_scalar)

    plt.scatter([x[0] for x in reconstruct_birds],
            [x[1] for x in reconstruct_birds])
    plt.show()

            

    # -   Reconstructed data + post-processing (mean, std)

def housing_plotter():
    x = fetch_california_housing