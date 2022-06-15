"""
Quick code for gp niching for multi-class classification

Need to pull out the parameters for this. Go line by line because there are so many of them currently. 

Could be same issues relating to replace=True rather than replace=False
"""



from code.data_processing import get_data
from code.learners.EC.GP import gp_member_generation
from code.member_selection.greedyEnsemble import greedyEnsemble
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

    

def get_toolbox(pset, t_size, max_depth, Xsubset, ysubset):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_calculation, toolbox=toolbox, X=Xsubset, y=ysubset) # HERE?
    toolbox.register("select", tools.selTournament, tournsize=t_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    return toolbox


def difference(pc1, pc2):
    dist = np.linalg.norm(pc1 - pc2)
    return dist 


def fitness_calculation(individual, toolbox, X, y):
    """
    Fitness function. Compiles GP then tests
    """
    func = toolbox.compile(expr=individual)
    ypred = GP_predict(func, X, np.unique(y)) # ave here? 
    #ypred = majority_voting(ypred)
    x = accuracy(y, ypred)
    return x,

# GP_predict(e1, X), GP_predict(e2, X)
def clearing_method(pop, toolbox, X, y, radius, capacity):
    sorted_ensemble = sorted(pop, key=lambda member : accuracy(y, GP_predict(toolbox.compile(expr=member), X, np.unique(y))), reverse=True) # DESCENDING 
    for i in range(len(sorted_ensemble)):
        if sorted_ensemble[i].fitness.values[0] > -np.inf:
            n = 0
            for j in range(i+1, len(pop), 1): # is this loop right? 
                ce1 = toolbox.compile(expr=sorted_ensemble[i])
                ce2 = toolbox.compile(expr=sorted_ensemble[j])
                if (sorted_ensemble[j].fitness.values[0] > -np.inf) and (difference(GP_predict(ce1, X, np.unique(y)), GP_predict(ce2, X, np.unique(y))) < radius):
                    if n < capacity:
                        n = n + 1
                    else:
                        sorted_ensemble[j].fitness.values = (-np.inf,)



def divnichegp_member_generation(X,y, params, seed):
    random.seed(seed)
    # unpack parameters
    max_depth = params["max_depth"]
    pc = params["pc"]
    pm = params["pm"]
    ngen = params["ngen"]
    p_size = params['p_size']
    verbose = params["verbose"]
    t_size = params['t_size']
    batch_size = params['batch_size']
    radius = params['radius']
    capacity = params['capacity']

    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # max
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Initalise tool box
    Xsubset = X
    ysubset = y
    toolbox = get_toolbox(pset, t_size, max_depth, Xsubset, ysubset)

    # Run GP
    pop = toolbox.population(n=p_size)

    halloffame = tools.HallOfFame(1)

    # Stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    df_data=[]
    batch_size = len(y)
    params["batch_size"] = len(y)
    # Generation 
    for gen in range(1, ngen + 1):
        # First take a sample from training 
        idx = np.random.choice(np.arange(len(X)), batch_size, replace=True)
        Xsubset = X[idx]
        ysubset = y[idx]

        print(f'gen = {gen}')
        # Produce offspring 
        offspring_a = toolbox.select(pop, len(pop))
        offspring_a = varAnd(offspring_a, toolbox, pc, pm)

        # Evaluate
        invalid_ind = [ind for ind in offspring_a if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(offspring_a)
        
        # Apply clearning method 
        clearing_method(offspring_a, toolbox, Xsubset, ysubset, radius=radius, capacity=capacity) # sets -inf to members to drop
        
        pop[:] = offspring_a
        record = mstats.compile(pop) if mstats else {}
        df_data.append(list(record['fitness'].values()) + list(record['size'].values()))

    # Apply greedy member selection algorithm
    return [toolbox.compile(ind) for ind in pop], pd.DataFrame(data=df_data), [str(ind) for ind in pop]
