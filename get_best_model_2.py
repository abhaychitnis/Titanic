import pandas as pd

def get_best_model(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    from sklearn.metrics import confusion_matrix, classification_report
    import importlib
    model_to_evaluate = pd.read_csv('classification_models_list.csv')
    for i in range(len(model_to_evaluate.index)):
        package = model_to_evaluate['package'][i]
        name = model_to_evaluate["name"][i]
        if pd.notnull(model_to_evaluate['function'][i]):
            function_name = model_to_evaluate['function'][i]
        else:
            function_name = ''

        print ('Scores for classification model, ', model_to_evaluate['model_name'][i], ' are : ')
        ic = getattr(__import__(package, fromlist=[name]), name)
        classifier = eval('ic('+function_name+')')
        classifier.fit(X_train, y_train)

# Predicting the Test set results
        y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        print (cr)
    return(0)
