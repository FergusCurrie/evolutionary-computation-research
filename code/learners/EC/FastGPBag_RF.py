
from code.data_processing import get_data
from code.learners.randomforest.randomforests import random_forest_classifier_member_generation
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
from code.decision_fusion.voting import majority_voting
from code.learners.EC.dtparse import SklearnParse
from scipy.spatial.distance import hamming

import itertools


def rf_init_generator(rf_pop, counter):
    value = next(counter)
    p = rf_pop[value]
    return p

def get_toolbox(pset, t_size, max_depth, X, y, curr_ensemble, rf_pop, counter, use_hamming):
    toolbox = base.Toolbox()
    #toolbox.register("expr", gp.fergus_genHalfAndHalf, pset=pset, min_=4, max_=max_depth, rf_pop=rf_pop)
    toolbox.register("expr", rf_init_generator, rf_pop=rf_pop, counter=counter)


    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", bagging_fitness_calculation, toolbox=toolbox, X=X, y=y, ensemble=curr_ensemble, use_hamming=use_hamming) # HERE?
    toolbox.register("select", tools.selTournament, tournsize=t_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    return toolbox


def difference(pc1, pc2, use_hamming):
    if use_hamming:
        return hamming(pc1, pc2)
    dist = np.linalg.norm(pc1 - pc2)
    return dist 

def bagging_fitness_calculation(individual, toolbox, X, y, ensemble, use_hamming):
    """
    Fitness function. Compiles GP then tests
    """
    e1 = toolbox.compile(expr=individual)
    temp_ensemble = ensemble + [e1] 
    # check difference 
    delta = np.inf
    for e2 in ensemble:
        d = difference(GP_predict(e1, X, np.unique(y)), GP_predict(e2, X, np.unique(y)), use_hamming)  # this uses the selection of the ensemble, think that is UCARP specific 
        if d < delta:
            delta = d
    if delta == 0:
        return -np.inf, 

    # calculate the temporary ensemble
    ypred = []
    for e in temp_ensemble:
        ypred.append(GP_predict(e, X, np.unique(y))) # might have to do one by one then combine
    ypred = np.array(ypred)
    assert(ypred.shape == (len(temp_ensemble),len(X)))
    ypred = majority_voting(ypred)
    return accuracy(y, ypred), # here



def rf_gp_member_generation(X,y, params, seed):
    random.seed(seed)
    # unpack parameters
    max_depth = params["max_depth"]
    pc = params["pc"]
    pm = params["pm"]
    ngen = params["ngen"]
    p_size = params['p_size']
    verbose = params["verbose"]
    t_size = params['t_size']
    curr_ensemble = params['ensemble']

    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # max
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Initalise tool box
    
    counter = itertools.count()

    parser = SklearnParse(pset)

    rf_params = {"n_estimators":p_size, "seed":seed, "max_depth":max_depth}
    rf_model = random_forest_classifier_member_generation(X, y, rf_params)


    n_unique = len(np.unique(y))
    rf_pop = parser.sklearn_random_forest_to_deap_gp_pop(rf_model, n_unique)
    
    toolbox = get_toolbox(pset, t_size, max_depth, X, y, curr_ensemble, rf_pop, counter, params['use_hamming'])



    # Run GP
    pop = toolbox.population(n=p_size)
    #print(type(pop))    # list 
    #print(type(pop[0])) # <class 'deap.creator.Individual'>      ....   This is good! 




    halloffame = tools.HallOfFame(1)

    # Stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    df_data = []


    for gen in range(1, ngen + 1):
        offspring_a = toolbox.select(pop, len(pop))
        offspring_a = varAnd(offspring_a, toolbox, pc, pm) # fails here currently 
        invalid_ind = [ind for ind in offspring_a if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(offspring_a)
        pop[:] = offspring_a

        # Logging 
        record = mstats.compile(pop) 
        df_data.append(list(record['fitness'].values()) + list(record['size'].values()))
    

    # nodes, edges, labels = gp.graph(pop[0])


    #return [toolbox.compile(ind) for ind in pop], df_data, [str(ind) for ind in pop]
    return toolbox.compile(halloffame[0]), np.array(df_data), str(halloffame[0])


#######################################################################################################################
# Bagging 
#######################################################################################################################



def gp_rf_bagging_member_generation(X, y, params, seed): # this is going to call the innergp a few times. 
    ncycles  = params['ncycles']
    batch_size = params['batch_size']
    ensemble = []
    ensemble_strings = []

    sum_history = np.ones((params['ngen'], 2))
    batch_size = len(y)
    params["batch_size"] = len(y)
    for c in range(ncycles):
        # evolve the ensemble for this cycle
        idx = np.random.choice(np.arange(len(X)), batch_size, replace=True)
        Xsubset = X[idx]
        ysubset = y[idx]
        params['ensemble'] = ensemble
        compiled_best, min_history, str_best = rf_gp_member_generation(Xsubset, ysubset, params, seed+c)
        sum_history += min_history

        ensemble.append(compiled_best) # complied lambda
        ensemble_strings.append(str_best) # str of member 
    


    return ensemble, pd.DataFrame(data=(sum_history/ncycles)), ensemble_strings # temporarily only saving the first 