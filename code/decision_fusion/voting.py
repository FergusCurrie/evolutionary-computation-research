import numpy as np


def binary_voting(ypred : np.array) -> np.array:
    """
    Take a 2D np array of (n_learners, n_datapoints) and average along learn axis.
    Threshold to 0/1 depending on if value is closer

    Args:
        ypred (np.array): 2d array of predictions

    Returns:
        np.array: 1d array of predictions with shape (n_datapoints)
    """
    avg = np.sum(ypred, axis=0) / ypred.shape[0]
    avg[avg >= 0.5] = 1
    avg[avg < 0.5] = 0
    return avg

def majority_voting(ypred : np.array):
    assert(ypred.shape[0] < ypred.shape[1])
    result = []
    for i in range(ypred.shape[1]):
        unique, counts = np.unique(ypred[:,i], return_counts=True, axis=0)
        result.append(unique[np.argmax(counts)])
    return np.array(result)


def weighted_voting(ypred : np.array, weights : np.array, binary=True) -> np.array:
    """
    Take a 2D np array of (n_learners, n_datapoints) and average along learn axis.
    Threshold to 0/1 depending on if value is closer

    Args:
        ypred (np.array): 2d array of predictions
        weights (np.array) : weighting for each gp classifier

    Returns:
        np.array: 1d array of predictions with shape (n_datapoints)
    """
    ypred = ypred * weights
    if binary:
        return binary_voting(ypred)
    return majority_voting(ypred)



def winner_takes_all(ypred : np.array):
    """_summary_

    Args:
        ypred (np.array): (n_learners, n_datapoints)

    Returns:
        _type_: _description_
    """
    return np.argmax(ypred, axis=0)

