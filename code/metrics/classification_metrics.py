import numpy as np

def calculate_confusion_matrix(y_true, y_pred):
    confusion_matrix = [[0,0],[0,0]] # first index is 1=true/0=false, second is 1=positive, 0=negative
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            if yp == 1:
                confusion_matrix[0][1] += 1
            if yp == 0:
                confusion_matrix[1][0] += 1
        elif yt == 1:
            if yp == 1:
                confusion_matrix[1][1] += 1
            elif yp == 0:
                confusion_matrix[0][0] += 1
    return confusion_matrix

def accuracy(cm):
    """confusion matrix # first index is 1=true/0=false, second is 1=positive, 0=negative """
    return (cm[1][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1])

def ensemble_voting(X,F):
    ypred = []
    for x in X:
        _yp = 0
        preds = np.array([single_pred(x, f) for f in F])
        if len(preds[preds == 0]) < len(preds[preds == 1]):
            _yp = 1
        ypred.append(_yp)
    return ypred

def single_pred(x, f): # thresholding
    yp = 1
    if f(*x) < 0:
        yp = 0
    return yp

def tpr(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][0])

def tnr(confusion_matrix):
    return confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][1])

def ave(confusion_matrix, w):
    return (w * tpr(confusion_matrix)) + ((1-w) * tnr(confusion_matrix))