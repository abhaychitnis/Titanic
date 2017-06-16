from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, \
                             accuracy_score

def best_model(X_train, X_test, y_train, y_test, eval_parm):

    if eval_parm == 'deep':
        parameters = {'n_neighbors': (4, 5, 7, 10), \
                      'weights': ('uniform', 'distance'), \
                      'algorithm': ('ball_tree', 'kd_tree', 'brute'), \
                      'metric': ('chebyshev', \
                                 'minkowski'), \
                      'p': (1, 2, 3, 4)
                      }
    elif eval_parm == 'test':
        parameters = {'n_neighbors': (4, 5), \
                      'weights': ('uniform', 'distance'), \
                      'algorithm': ('ball_tree', 'kd_tree', 'brute'), \
                      'metric': ('chebyshev', \
                                 'minkowski'), \
                      'p': (2, 3)
                      }
    
    classifier = KNeighborsClassifier()

    gs = GridSearchCV(classifier, parameters)
    gs.fit(X_train, y_train)

# Predicting the Test set results
    y_pred = gs.predict(X_test)


# Making the Confusion Matrix
    m_precision, m_recall, m_fbeta, _ = \
        precision_recall_fscore_support(y_test, y_pred, average='macro')
    m_score = accuracy_score(y_test, y_pred)


    model_eval_param = {"accuracy_score": m_score, "precision": m_precision, \
                        "recall": m_recall, "fbeta": m_fbeta, \
                        "model_sign": gs.best_estimator_}

    return (model_eval_param)
