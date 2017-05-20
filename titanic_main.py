
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


preprocessing_override = pd.read_csv('preprocessing_override.csv')

dataset_X = pd.read_csv('train.csv')
dataset_y = dataset_X['Survived']

dataset_X_verify = pd.read_csv('test.csv')

del dataset_X['Survived']
del preprocessing_override['Survived']

import preprocess_data as prd

preprocessed_data = prd.preprocess_data(dataset_X, dataset_y, preprocessing_override, dataset_X_verify)

print (preprocessed_data["X"][0:3,:])
