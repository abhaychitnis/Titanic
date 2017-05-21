from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def best_model(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

# Predicting the Test set results
    y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    model_eval_param = {"cnf_matrix": cm, "class_report": cr}

    return (model_eval_param)