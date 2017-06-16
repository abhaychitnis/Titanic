from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, \
                             accuracy_score

def best_model(X_train, X_test, y_train, y_test, eval_parm):

    if eval_parm == 'deep':
        parameters = {'n_estimators': (5, 10, 15, 20, 25, 30, 50, 100), \
                 'learning_rate': (0.01, 0.1, 0.2, 0.5, 0.7, 0.9), \
                 'max_depth': (2, 3, 4), \
                 'min_samples_split': (2, 3), \
                 'min_samples_leaf': (2, 3), \
                 'max_features': ('auto', 'log2', None) \
                 }
    elif eval_parm == 'test':
        parameters = {'n_estimators': (200, 300), \
                 'learning_rate': (0.1, 0.2, 0.3), \
                 'min_samples_leaf': (3, 4)
                      }


    classifier = GradientBoostingClassifier()

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
