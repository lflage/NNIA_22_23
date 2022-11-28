from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import mean_squared_error

dataset = load_diabetes()
test_splits = [0.1,0.5,0.9]
for test_size in test_splits:
    # The input data are in `dataset.data`, targets are in `dataset.target`.
    #print(dataset.data.shape, dataset.target.shape)
    # If you want to learn more about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.
    #print(dataset.DESCR)

    # TODO: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    column_ones = np.ones((len(dataset.data),1))
    new_data = np.append(dataset.data, column_ones,axis=1)
    #print(new_data.shape)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size, random_state=seed`.
    data_train, data_test, y_train, y_test = train_test_split(new_data,
        dataset.target, test_size=test_size, random_state=42)

    data_train = np.matrix(data_train)
    data_test = np.matrix(data_test)


    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    w = np.linalg.inv(data_train.T @ data_train) @ data_train.T @ y_train
    
    # Extracting the Bias from the weights matrix
    bias = w[0,-1]
    w = w[0,:10]

    # TODO: Predict target values on the test set.
    y_pred = data_test[:,:10] @ w.T + bias

    #print(y_pred.A1.shape)
    #print(y_test.shape)
    # TODO: Manually compute root mean square error on the test set predictions.
    m = len(y_train)
    error = y_test - y_pred.A1 
    sq_error = error ** 2
    sum_sq_error = np.sum(sq_error)
    rmse = sum_sq_error/m
    #rmse = np.sum((y_pred.A1-y_test)**2)/m
    scikit_mse = mean_squared_error(y_pred.A1,y_test)

    print('my_rmse:', rmse)
    print('scikit:', scikit_mse)

    # mean_squared_error