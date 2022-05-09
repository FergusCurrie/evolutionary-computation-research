"""
From an ensemble selec
From this paper : https://dl.acm.org/doi/10.1145/1276958.1277317

"""
from code.metrics.classification_metrics import *
from typing import Callable


def rpf(ensemble : list, X : np.array, y : np.array, decision_fusion_func : Callable) -> list:
    """RPF is a naive ensemble selection algorithm. We iteratively search through the ensemble, keeping only 
    individual models with more than 50% accuracy. 

    Args:
        ensemble (list): list of model objects 
        X (np.array): dataset
        y (np.array): targets 
        decision_fusion_func (Callable): which decision fusion function to apply

    Returns:
        list: list of model objects in the final ensmble
    """
    # First sort population 
    sorted_ensemble = sorted(ensemble, key=lambda member : accuracy(y, member.predict(X, np.unique(y))), reverse=True) # DESCENDING 

    final_ensemble = []
    for e in sorted_ensemble:
        ypred = decision_fusion_func(e.predict(X, np.unique(y)))
        acc = accuracy(y, ypred)
        if acc > e:
            final_ensemble.append(e)
        else:
            return final_ensemble
