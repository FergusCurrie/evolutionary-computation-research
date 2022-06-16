
from code.data_processing import get_data
from code.learners.EC.GP import gp_member_generation
from code.metrics.classification_metrics import *
from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
import operator
import random
from code.metrics.classification_metrics import *
from code.learners.EC.deap_extra import GP_predict, get_pset
import pandas as pd 
from code.decision_fusion.voting import binary_voting
from code.learners.EC.m3gp.M3GP import M3GP


def difference(pc1, pc2):
    dist = np.linalg.norm(pc1 - pc2)
    return dist 

def bagging_fitness_calculation(individual, toolbox, X, y, ensemble):
    """
    Fitness function. Compiles GP then tests
    """
    e1 = toolbox.compile(expr=individual)
    temp_ensemble = ensemble + [e1] 
    # check difference 
    delta = np.inf
    for e2 in ensemble:
        d = difference(GP_predict(e1, X, np.unique(y)), GP_predict(e2, X, np.unique(y)))  # this uses the selection of the ensemble, think that is UCARP specific 
        if d < delta:
            delta = d
    if delta == 0:
        return np.inf, 

    # calculate the temporary ensemble
    ypred = []
    for e in temp_ensemble:
        ypred.append(GP_predict(e, X, np.unique(y))) # might have to do one by one then combine
    ypred = np.array(ypred)
    assert(ypred.shape == (len(temp_ensemble),len(X)))
    ypred = binary_voting(ypred)
    return accuracy(y, ypred), # here


#######################################################################################################################
# Bagging 
#######################################################################################################################

def m3gpbag_member_generation(X, y, params, seed): # this is going to call the innergp a few times. 
    ncycles  = params['ncycles']
    batch_size = params['batch_size']
    ensemble = []
    es = [] # strings 
    dfs = []
    batch_size = len(y)
    params["batch_size"] = len(y)
    for c in range(ncycles):
        # evolve the ensemble for this cycle
        idx = np.random.choice(np.arange(len(X)), batch_size, replace=True)
        Xsubset = X[idx]
        ysubset = y[idx]
        params['ensemble'] = ensemble

        Tr_X = pd.DataFrame(data=Xsubset)    # frame for x
        Tr_Y = pd.DataFrame(data=ysubset)[0] # series for y
        
        m3gp = M3GP(population_size = params['p_size'], max_generation = params['p_size'], verbose=False, limit_depth = params['max_depth'])
        m3gp.fit(Tr_X, Tr_Y)

        best_individual = m3gp.getBestIndividual()

        ensemble.append(best_individual) 
        es.append(str(best_individual))
    


    return ensemble, None, '' # temporarily only saving the first 

