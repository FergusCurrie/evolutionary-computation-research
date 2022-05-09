"""


"""
from code.metrics.classification_metrics import *
from typing import Callable

def distance(e1, e2, X, y):
    pc1 = e1.predict(X, np.unique(y))
    pc2 = e2.predict(X, np.unique(y))
    dist = np.linalg.norm(pc1 - pc2)
    return dist 

def greedyEnsemble(ensemble : list, X : np.array, y : np.array, decision_fusion_func : Callable, params) -> list:
    gamma = params['radius']

    # First sort population 
    sorted_ensemble = sorted(ensemble, key=lambda member : accuracy(y, member.predict(X, np.unique(y))), reverse=True) # DESCENDING 
    selected_ensemble = []
    fitness_selected_ensemble = np.inf

    for i in range(len(sorted_ensemble)-1):
        inNiche = False
        if selected_ensemble == []:
            pass
        else:
            for e1 in sorted_ensemble:
                if distance(e1, sorted_ensemble[i], X, y) < gamma:
                    inNiche = True
                    break
        
        if not inNiche:
            ens_temp = selected_ensemble + [sorted_ensemble[i]]
            # calculate fitness of the temp ensemble 
            ypred = []
            for e in ens_temp:
                ypred.append(e.predict(X, np.unique(y))) # might have to do one by one then combine
            ypred = np.array(ypred)
            assert(ypred.shape == (len(ens_temp),len(X)))
            ypred = decision_fusion_func(ypred)
            fitness_temp = accuracy(y, ypred)
            # if this is an improvement, update.
            if fitness_temp < fitness_selected_ensemble:
                selected_ensemble = ens_temp
                fitness_selected_ensemble = fitness_temp

    return selected_ensemble