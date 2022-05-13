from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# https://scikit-learn.org/stable/modules/ensemble.html#forest





def random_forest_classifier_member_generation(X, y, params):
    assert(params['n_estimators'] != None)
    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'], 
        random_state = params['seed'],
        max_depth=params["max_depth"]
    )
    clf.fit(X, y)
    return clf


def adaboost_classifier_member_generation(X, y, params):
    n_estimators = params['n_estimators']
    assert(n_estimators != None)
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    return clf

def evaluate_sklearn(clf, X):
    return clf.predict(X)





