# Code for MOGP 
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

def get_stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", np.min)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    return mstats, logbook

def get_toolbox(pset, params, X,y):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=params['max_depth'])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("species", tools.initRepeat, list, toolbox.individual, params['species_size'])
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_calculation, toolbox=toolbox, X=X, y=y) # HERE?
    toolbox.register("select", tools.selTournament, tournsize=params['t_size'])
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=params['max_depth'])
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params['max_depth']))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params['max_depth']))
    toolbox.register("get_best", tools.selBest, k=1)
    return toolbox

def fitness_calculation(pop, toolbox, X, y, w=0.5):
    """
    Fitness function. Compiles GP then tests
    """
    funcs = [toolbox.compile(expr=ind) for ind in pop]
    ypreds = np.array([GP_predict(func, X, np.unique(y)) for func in funcs])
    votes = majority_voting(ypreds)
    x = accuracy(votes, y)
    return x,

def ccgp_member_generation(X,y, params, seed):
    random.seed(seed)


    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # max
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)  

    # Initalise tool box
    toolbox = get_toolbox(pset, params, X, y)

    # Run GP
    #pop = toolbox.population(n=params['p_size'])
    species = [toolbox.species() for _ in range(params['nspecies'])]
    representatives = [random.choice(species[i]) for i in range(params['nspecies'])]

    # Stats
    mstats, logbook = get_stats()
    df_data = []

    for gen in range(1, params['ngen'] + 1):
        print(f'Current generation is : {gen}')
        # Initialize a container for the next generation representatives
        next_repr = [None] * len(species)
        for i, s in enumerate(species): # i is the species index, s is the species 
            # s is the sub population, we deal with it equivlantish to population
            # Vary the species individuals
            s = varAnd(s, toolbox, params['pc'], params['pm'])

            # Get the representatives excluding the current species
            r = representatives[:i] + representatives[i+1:] 
            for ind in s:
                ind.fitness.values = toolbox.evaluate([ind] + r) # Add the individual to representatives

            # Select the individuals
            species[i] = toolbox.select(s, len(s))  # Tournament selection
            next_repr[i] = toolbox.get_best(s)[0]   # Best selection

        representatives = next_repr # representatives are 10 best. 
        record = mstats.compile(representatives) if mstats else {}
        df_data.append(list(record['fitness'].values()) + list(record['size'].values()))

    return [toolbox.compile(ind) for ind in representatives], pd.DataFrame(data=df_data), [str(ind) for ind in representatives]

