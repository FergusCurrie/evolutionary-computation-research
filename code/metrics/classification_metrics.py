import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def error_rate(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


"""def ensemble_voting(X, F):
    ypred = []
    for x in X:
        _yp = 0
        preds = np.array([single_pred(x, f) for f in F])
        if len(preds[preds == 0]) < len(preds[preds == 1]):
            _yp = 1
        ypred.append(_yp)
    return ypred"""

def calculate_confusion_matrix(y_true, y_pred, verbose=False):
    return confusion_matrix(y_true, y_pred)


def tpr(cm):
    if len(cm.ravel()) == 0: # check for per class case where none of a class exists
        return 0
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn)


def tnr(cm):
    if len(cm.ravel()) == 0: # check for per class case where none of a class exists
        return 0
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)


def ave(cm, w):
    return (w * tpr(cm)) + ((1 - w) * tnr(cm))


def binary_metric(ytrue, ypred):
    # full
    full_cm = calculate_confusion_matrix(ytrue, ypred)
    tn, fp, fn, tp = full_cm.ravel()
    full_acc = accuracy(ytrue, ypred)
    majority_acc = accuracy(ytrue[ytrue == 1], ypred[ytrue == 1])
    minority_acc = accuracy(ytrue[ytrue == 0], ypred[ytrue == 0])

    return [full_acc, majority_acc, minority_acc, tn, fp, fn, tp]
