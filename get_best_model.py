import pandas as pd

def get_best_model(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    import importlib
    model_to_evaluate = pd.read_csv('classification_models_list.csv')
    for i in range(len(model_to_evaluate.index)):
        print ('Scores for classification model, ', model_to_evaluate['model_name'][i], ' are : ')
    
        brf = importlib.import_module(model_to_evaluate['file_name'][i])
        model_param = brf.best_model(X_train, X_test, y_train, y_test)
        print (model_param["class_report"])

    return(0)
