import pandas as pd
from sklearn.externals import joblib
import pickle

def get_best_model(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    import importlib

    return_list =[]
    model_to_evaluate = pd.read_csv('classification_models_list.csv')
    for i in range(len(model_to_evaluate.index)):
    
        brf = importlib.import_module(model_to_evaluate['file_name'][i])
        eval_parm = model_to_evaluate['eval_parm'][i]
        model_param = brf.best_model(X_train, X_test, y_train, y_test, eval_parm)

        model_param["model_name"] = model_to_evaluate['model_name'][i]

        return_list.append(model_param)

    best_model_found = evalModelList(X_train, y_train, return_list)
 
    return(pickle.dumps(best_model_found))

def evalModelList (X_train, y_train, return_list):

    modelStore = pd.DataFrame(return_list)

    bestModelRow = modelStore.ix[modelStore['fbeta'].idxmax()]

    best_model_found = bestModelRow['model_sign']

    return(best_model_found)
    
