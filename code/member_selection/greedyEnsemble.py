"""


"""
from code.decision_fusion.voting import binary_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import *
from typing import Callable

def distance(e1, e2):
    return 0

def greedyEnsemble(ensemble : list, X : np.array, y : np.array, decision_fusion_func : Callable) -> list:

    gamma = 1

    # First sort population 
    sorted_ensemble = sorted(ensemble, key=lambda member : accuracy(y, GP_predict(member, X)), reverse=True) # DESCENDING 
    selected_ensemble = []
    fitness_selected_ensemble = np.inf

    for i in range(len(sorted_ensemble)-1):
        inNiche = False
        for e1 in sorted_ensemble:
            if distance(e1, sorted_ensemble[i]) < gamma:
                inNiche = True
        
        if not inNiche:
            ens_temp = selected_ensemble + sorted_ensemble[i]
            # calculate fitness of the temp ensemble 
            ypred = []
            for e in ens_temp:
                ypred.append(GP_predict(e, X)) # might have to do one by one then combine
            ypred = np.array(ypred)
            assert(ypred.shape == (len(ens_temp),len(X)))
            ypred = binary_voting(ypred)
            fitness_temp = accuracy(y, ypred)
            # if this is an improvement, update.
            if fitness_temp < fitness_selected_ensemble:
                selected_ensemble = ens_temp
                fitness_selected_ensemble = fitness_temp

    return selected_ensemble