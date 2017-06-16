from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, \
                    accuracy_score

def best_model(X_train, X_test, y_train, y_test, eval_parm):

    if eval_parm == 'deep':
        parameters = {'C': (1.0, 2.0), \
                 'kernel': ('linear', 'rbf', 'poly'), \
                 'degree': (2, 3)
                      }
    elif eval_parm == 'test':
        parameters = {'C': (1.0, 2.0), \
                 'kernel': ('linear', 'rbf', 'poly'), \
                 'degree': (2, 3)
                      }
    
    classifier = SVC(random_state = 0)

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
