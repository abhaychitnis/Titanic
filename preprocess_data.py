import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def preprocess_data(dataset_X, dataset_y, override, dataset_X_verify):

# common parameters

    missing_values_not_applicable = 0
    missing_values_drop_rows = 1
    missing_values_fill_mean = 2
    missing_values_drop_column = 3
    missing_values_not_decided = 4

    import common_functions as cm

    preprocess_list = cm.preprocess_ind(dataset_X, override)

    category_encoding_columns = preprocess_list[0]
    missing_values_strategy = preprocess_list[1]
    drop_strategy_columns = preprocess_list[2]
    normalize_strategy_columns = preprocess_list[3]


    y = dataset_y.iloc[:].values
    X = dataset_X.iloc[:, :].values
    X_verify = dataset_X_verify.iloc[:, :].values

    X_y_missing_values_managed = cm.manage_missing_values(X, y, missing_values_strategy)
    X = X_y_missing_values_managed['X']
    y = X_y_missing_values_managed['y']

    X = cm.manage_normalize_values(X, normalize_strategy_columns)
    
    X = cm.manage_category_encoding(X, category_encoding_columns, drop_strategy_columns)

    return_y_object = cm.manage_target_values(y)

    y = return_y_object['y']
    standardScaler_y = return_y_object['standardScaler_y']
    labelEncoder_y = return_y_object['labelEncoder_y']

    if labelEncoder_y == None:
        print ("No Label Encoding of Y")
    if standardScaler_y == None:
        print ("No Standard Scaling of Y")

  
    return ({"X":X, "y":y})
