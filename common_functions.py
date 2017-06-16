import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_ind (dataset, override):

    unique_identification_cutoff = 0.1
    drop_column_cutoff = 0.75

    override_ind = 0
    encoding_override_ind = 1
    dropcolumn_override_ind = 2
    normalize_override_ind = 3

    num_rows, num_columns = dataset.shape
    
    null_list = dataset.isnull().sum()
    encoding_override = list((override == encoding_override_ind).values)[0]
    drop_column_override = list((override == dropcolumn_override_ind).values)[0]
    normalize_column_override = list((override == normalize_override_ind).values)[0]

    return_list = []
    category_encoding_columns = []
    missing_value_strategy = []
    drop_strategy_columns = []
    normalize_strategy_columns = []

#Inspect every columns
    for i in range(num_columns):
# Inspect uniquenes

#   Make sure that the null values are not considered while finding out the count of unique values for the column
        presence_of_missing_values = False
        percentage_missing_values = 100*null_list[i]/num_rows
        column_with_notnull_values = dataset[dataset.columns[i]][dataset[dataset.columns[i]].notnull()==True]
        if null_list[i] != 0:
            presence_of_missing_values = True

#   Initialize indicators
        drop_ind = False
        encode_ind = False

# disparate_data_index is the ratio of number of unique values in the column to the number of rows. Lower values
# indicates potential of uniqueness. A value of 1 indicates that every value in the coulmn is different than the rest
        disparate_data_index = ((len(column_with_notnull_values.unique())) /(num_rows-null_list[i]))


# Inspect the data type
        number_datatype = False
        if dataset[dataset.columns[i]].dtype in ('int64', 'float64'):
            number_datatype = True

        if presence_of_missing_values:
            if percentage_missing_values > 50:
#Set the missing value strategy to removing the column(or feature)
                missing_value_strategy.append(3)
                drop_strategy_columns.append(i)
                drop_ind = True
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

        if disparate_data_index < drop_column_cutoff   \
        or number_datatype \
        or drop_column_override[i]  \
        or missing_value_strategy[i] == 3:
            pass
        else:
            drop_strategy_columns.append(i)
            drop_ind = True

# Determine if the column is a candidate for feature encoding
        if disparate_data_index > unique_identification_cutoff   \
        or encoding_override[i]   \
        or drop_ind   \
        or missing_value_strategy[i] == 3:
            pass
        else:
            category_encoding_columns.append(i)
            encode_ind = True

# Determine if the column values need to be normalized

        if encode_ind or  \
           missing_value_strategy [i] == 3   or  \
           drop_ind  or \
           normalize_column_override[i] or \
           not number_datatype:
            pass
        else:
            normalize_strategy_columns.append(i)



    return_list.append(category_encoding_columns)
    return_list.append(missing_value_strategy)
    return_list.append(drop_strategy_columns)
    return_list.append(normalize_strategy_columns)

    return (return_list)

def manage_missing_values(X,y, missing_values_strategy):

# common parameters

    missing_values_not_applicable = 0
    missing_values_drop_rows = 1
    missing_values_fill_mean = 2
    missing_values_drop_column = 3
    missing_values_not_decided = 4

 # Taking care of missing data by dropping the rows

    missing_columns_drop_rows = list(np.where(np.array(missing_values_strategy) == missing_values_drop_rows))[0]

    for i in missing_columns_drop_rows:
        indices_of_empty_rows = np.where((pd.isnull(X[:,i]) == True))[0]

        X = np.delete(X, indices_of_empty_rows , axis=0)
        if y != None:
            y = np.delete(y, indices_of_empty_rows , axis=0)

# Taking care of missing data by filling with mean values

    missing_columns_tobe_filled_with_mean = list(np.where(np.array(missing_values_strategy) == missing_values_fill_mean))[0]

    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

    for i in missing_columns_tobe_filled_with_mean:
        imputer = imputer.fit(X[:, i:i+1])
        X[:, i:i+1] = imputer.transform(X[:, i:i+1])

    return ({"X":X, "y":y})

def manage_normalize_values(X, normalize_strategy_columns):

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    for i in normalize_strategy_columns:
        X[:, [i]] = sc_X.fit_transform(X[:, i].reshape(-1, 1))

    return(X)

def manage_target_values(y):

    unique_identification_cutoff_y = 0.01
    lc_y = None
    sc_y = None

    disparate_data_index_y = ((len(np.unique(y))) /len(y))

    
    if y.dtype in ('int64', 'float64'):
        number_datatype_y = True

    if number_datatype_y:
        if disparate_data_index_y > unique_identification_cutoff_y:
            from sklearn.preprocessing import StandardScaler
            sc_y = StandardScaler()
            y = sc_y.fit_transform(y.reshape(-1, 1))                             
    else:
        from sklearn.preprocessing import LabelEncoder
        lc_y = LabelEncoder()
        y = lc_y.fit_transform(y)

    labelEncoder_y = lc_y
    standardScaler_y = sc_y
        
    y_return = {"y": y, \
                "labelEncoder_y": labelEncoder_y, \
                "standardScaler_y": standardScaler_y}
    return(y_return)

def manage_category_encoding(X, category_encoding_columns, drop_strategy_columns):

# For unique string column that can be cosidered as classification, convert into classification encoding using onecode

    X_extract_encoded = None
    if not (not category_encoding_columns):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder_X = LabelEncoder()

        for i in (category_encoding_columns):
            X[:, i] = labelencoder_X.fit_transform(X[:, i])


        X_extract = X[:,category_encoding_columns]
        onehotencoder = OneHotEncoder(categorical_features = 'all')
        X_extract_encoded = onehotencoder.fit_transform(X_extract).toarray()

# Remove one of the encoded columns from each encoded category
        remove_columns = [0]
        for i in range (1, len(onehotencoder.n_values_)):
            remove_columns.append(remove_columns[i-1] + \
                                  onehotencoder.n_values_[i])

        X_extract_encoded = np.delete(X_extract_encoded, \
                      remove_columns, axis=1)
        
# Drop all marked columns, including the ones marked for encoding        
    X = np.delete(X, category_encoding_columns+drop_strategy_columns , axis=1)

# Append the encoded columns
    if X_extract_encoded == None:
        pass
    else:
        X = np.c_[X_extract_encoded, X]

    return(X)
