
# %load titanic-pre.py
# Data Preprocessing Template


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# common parameters

missing_values_not_applicable = 0
missing_values_drop_rows = 1
missing_values_fill_mean = 2
missing_values_drop_column = 3
missing_values_not_decided = 4

# Importing the dataset
dataset_complete = pd.read_csv('train.csv')
preprocessing_override = pd.read_csv('preprocessing_override.csv')
dataset_X = dataset_complete
dataset_y = dataset_complete['Survived']

del dataset_X['Survived']
del preprocessing_override['Survived']

def preprocess_ind (dataset, override):

    unique_identification_cutoff = 0.01
    drop_column_cutoff = 0.75

    override_ind = 0
    encoding_override_ind = 1
    dropcolumn_override_ind = 2
    
    num_rows, num_columns = dataset.shape
    null_list = dataset.isnull().sum()
    encoding_override = list((override == encoding_override_ind).values)[0]
    drop_column_override = list((override == dropcolumn_override_ind).values)[0]

    return_list = []
    category_encoding = []
    missing_value_strategy = []
    drop_column_strategy = []
    normalize_strategy = []

#Inspect every columns
    for i in range(num_columns):
# Inspect uniquenes 

#   Make sure that the null values are not considered while finding out the count of unique values for the column
        presence_of_missing_values = False
        percentage_missing_values = 100*null_list[i]/num_rows
        column_with_notnull_values = dataset[dataset.columns[i]][dataset[dataset.columns[i]].notnull()==True]
        if null_list[i] != 0:
            presence_of_missing_values = True
    
# disparate_data_index is the ratio of number of unique values in the column to the number of rows. Lower values
# indicates potential of uniqueness. A value of 1 indicates that every value in the coulmn is different than the rest
        disparate_data_index = ((len(column_with_notnull_values.unique())) /(num_rows-null_list[i]))

# Determine if the column is a candidate for feature encoding
        if disparate_data_index > unique_identification_cutoff: 
            category_encoding.append(False)
        elif encoding_override[i]:
            category_encoding.append(False)
        else:
            category_encoding.append(True)
# Inspect the data type
        number_datatype = False
        if dataset[dataset.columns[i]].dtype in ('int64', 'float64'):
            number_datatype = True
    
        if presence_of_missing_values:
            if percentage_missing_values > 50:
#Set the missing value strategy to removing the column(or feature)
                missing_value_strategy.append(3)
            elif percentage_missing_values < 5:
#Set the missing value strategy to removing the rows with missing value
                missing_value_strategy.append(1)
            elif number_datatype:
#Set the missing value strategy to setting the value to mean of the column values
                missing_value_strategy.append(2)
            else:
#Set the missing value strategy to UNKNOWN. This is related to non-numeric fields
                missing_value_strategy.append(4)
        else:
#Set the missing value strategy to not applicable as there are no missing values
            missing_value_strategy.append(0)
    
        if disparate_data_index < drop_column_cutoff:
            drop_column_strategy.append(False)
        elif drop_column_override[i]:
            drop_column_strategy.append(False)
        else:
            drop_column_strategy.append(True)
        
        
        if category_encoding[i]:
            normalize_strategy.append(False)
        elif missing_value_strategy == 3:
            normalize_strategy.append(False)
        elif drop_column_strategy[i]:
            normalize_strategy.append(False)
        elif number_datatype:
            normalize_strategy.append(True)
        else:
            normalize_strategy.append(False)
            
    return_list.append(category_encoding)
    return_list.append(missing_value_strategy)
    return_list.append(drop_column_strategy)
    return_list.append(normalize_strategy)
    
    return (return_list)
    
preprocess_list = preprocess_ind(dataset_X, preprocessing_override)

category_encoding_ind = preprocess_list[0]
missing_values_strategy = preprocess_list[1]
drop_column_strategy = preprocess_list[2]
normalize_strategy = preprocess_list[3]


y = dataset_y.iloc[:].values
X = dataset_X.iloc[:, :].values
X_test = pd.read_csv('test.csv').iloc[:,:].values

# Taking care of missing data by dropping the rows

missing_columns_drop_rows = list(np.where(np.array(missing_values_strategy) == missing_values_drop_rows))[0]

for i in missing_columns_drop_rows:
    indices_of_empty_rows = np.where((pd.isnull(X[:,i]) == True))[0]
    
    X = np.delete(X, indices_of_empty_rows , axis=0)
    y = np.delete(y, indices_of_empty_rows , axis=0)

    indices_of_empty_rows_test = np.where((pd.isnull(X_test[:,i]) == True))[0]
    X_test = np.delete(X_test, indices_of_empty_rows_test , axis=0)

    
    

# Taking care of missing data by filling with mean values

missing_columns_tobe_filled_with_mean = list(np.where(np.array(missing_values_strategy) == missing_values_fill_mean))[0]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

for i in missing_columns_tobe_filled_with_mean:
    imputer = imputer.fit(X[:, i:i+1])
    X[:, i:i+1] = imputer.transform(X[:, i:i+1])
    
    imputer = imputer.fit(X_test[:, i:i+1])
    X_test[:, i:i+1] = imputer.transform(X_test[:, i:i+1])


# Taking care of missing data by dropping columns

missing_columns_drop_column = list(np.where(np.array(missing_values_strategy) == missing_values_drop_column))[0]

# Delete columns from X matching the index numbers in missing_columns_drop_column

X = np.delete(X, missing_columns_drop_column , axis=1)
X_test = np.delete(X_test, missing_columns_drop_column , axis=1)


# Delete items from missing_values_strategy corresponding to the columns dropped in X

missing_values_strategy = list(np.delete(np.array(missing_values_strategy), missing_columns_drop_column , axis=0))

# Delete items from category_encoding_ind corresponding to the columns dropped in X

category_encoding_ind = list(np.delete(np.array(category_encoding_ind), missing_columns_drop_column , axis=0))

# Delete items from drop_column_strategy corresponding to the columns dropped in X

drop_column_strategy = list(np.delete(np.array(drop_column_strategy), missing_columns_drop_column , axis=0))

# Delete items from normalize_strategy corresponding to the columns dropped in X
normalize_strategy = list(np.delete(np.array(normalize_strategy), missing_columns_drop_column , axis=0))



# Drop columns corresponding to the columns marked in preprocessing as they are not likely to be relevant

drop_column = list(np.where(np.array(drop_column_strategy) == True))[0]

#print (drop_column)

# Delete columns from X matching the index numbers in missing_columns_drop_column

X = np.delete(X, drop_column , axis=1)
X_test = np.delete(X_test, drop_column , axis=1)

# Delete items from missing_values_strategy corresponding to the columns dropped in X

missing_values_strategy = list(np.delete(np.array(missing_values_strategy), drop_column , axis=0))

# Delete items from category_encoding_ind corresponding to the columns dropped in X

category_encoding_ind = list(np.delete(np.array(category_encoding_ind), drop_column , axis=0))

# Delete items from normalize_strategy corresponding to the columns dropped in X

normalize_strategy = list(np.delete(np.array(normalize_strategy), drop_column , axis=0))

# Delete items from drop_column_strategy corresponding to the columns dropped in X

drop_column_strategy = list(np.delete(np.array(drop_column_strategy), missing_columns_drop_column , axis=0))

if not drop_column_strategy:
    print ("something wrong !")

# For unique string column that can be cosidered as classification, convert into classification encoding using onecode

category_encoding_columns = list(np.where(np.array(category_encoding_ind) == True))[0]

if (category_encoding_columns.any()):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()

    for i in (category_encoding_columns):
        X[:, i] = labelencoder_X.fit_transform(X[:, i])
        
        X_test_drop_rows = np.where((pd.isnull(X_test[:,i]) == True))[0]
        if (X_test_drop_rows.any()):
            X_test = np.delete(X_test, X_test_drop_rows , axis=0)
        X_test[:,i] = labelencoder_X.transform(X_test[:, i])

    X_extract = X[:,category_encoding_columns]
    X_test_extract = X_test[:,category_encoding_columns]
    onehotencoder = OneHotEncoder(categorical_features = 'all')    
    X_extract_encoded = onehotencoder.fit_transform(X_extract).toarray()
    X_test_extract_encoded = onehotencoder.fit_transform(X_test_extract).toarray()

    X = np.delete(X, category_encoding_columns , axis=1)
    X_test = np.delete(X_test, category_encoding_columns , axis=1)
    category_encoding_ind = list(np.delete(np.array(category_encoding_ind), category_encoding_columns , axis=0))
# Delete items from normalize_strategy corresponding to the columns dropped in X
    normalize_strategy = list(np.delete(np.array(normalize_strategy), category_encoding_columns , axis=0))

    
    X = np.c_[X, X_extract_encoded]
    X_test = np.c_[X_test, X_test_extract_encoded]

# For numeric columns, scale the values appropriately

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
normalize_strategy_columns = list(np.where(np.array(normalize_strategy) == True))[0]
for i in normalize_strategy_columns:

    X[:, [i]] = sc_X.fit_transform(X[:, i].reshape(-1, 1))
    X_test_drop_rows = np.where((pd.isnull(X_test[:,i]) == True))[0]
    if (X_test_drop_rows.any()):
        X_test = np.delete(X_test, X_test_drop_rows , axis=0)

    X_test[:,[i]] = sc_X.transform(X_test[:,i].reshape(-1,1))

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
print ("All OK")