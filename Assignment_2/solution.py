import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
    mse = mean_squared_error(birds,reconstruct_birds)

    return("MSE: {}".format(mse))