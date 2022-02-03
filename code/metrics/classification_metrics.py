import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


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
    full_acc = accuracy(ytrue, ypred)
    full_tpr = tpr(full_cm)
    full_tnr = tnr(full_cm)

    return [full_acc, full_tpr, full_tnr]
