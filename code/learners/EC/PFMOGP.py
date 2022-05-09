from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
from deap import algorithms
import operator
from code.learners.EC.deap_extra import GP_predict, get_pset
from code.metrics.classification_metrics import accuracy
from code.learners.EC.my_nsga2 import selncl
import pandas as pd 
import random

# https://github.com/DEAP/deap/blob/master/doc/api/tools.rst


def get_toolbox(pset, max_depth, X, y):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_calculation, toolbox=toolbox, X=X, y=y)  # HERE?
    toolbox.register("select", selncl, X=X, y=y, div='pf', toolbox=toolbox)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    return toolbox

def fitness_calculation(individual, toolbox, X, y):
    """
    Fitness function. Compiles tree then calls mse function
    """
    func = toolbox.compile(expr=individual)

    # Calculate accuracy on class 0
    class_0_acc = accuracy(y[y==0], GP_predict(func, X[y == 0], np.unique(y)))
    class_1_acc = accuracy(y[y==1], GP_predict(func, X[y == 1], np.unique(y)))

    return class_0_acc, class_1_acc,



def pfmo_member_generation(X, y, params, seed):
    """Generation an ensemble of GP trees using multi-objective tournament selection. 

    Args:
        X (np.array): training data 
        y (np.array): training labels 

        PARAMS DICT
        p_size (int): number of individuals in population
        max_depth (int): max depth of gp tree. limits program size
        pc (float): probability of individual crossing
        pm (float): probability of individual mutating
        ngen (int): number of generations before termination
        verbose (bool, optional): whether to print progress of training or not 

    Returns:
        [list]: ensemble members... which are just lambda functions 
    """
    random.seed(seed)
   

    max_depth = params["max_depth"]
    pc = params["pc"]
    pm = params["pm"]
    ngen = params["ngen"]
    verbose = params["verbose"]
    p_size = params["p_size"]
    print(f'psize={p_size}')

    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * 2)  # maximisation * n_objectives
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # Initalise tool box
    toolbox = get_toolbox(pset, max_depth, X, y)

    # Run GP
    pop = toolbox.population(n=p_size)

    # Stats
    mstats = tools.Statistics(lambda ind: ind.fitness.values)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"


    # Init pop fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = mstats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    #print(logbook.stream)


    # Evolution process 
    for gen in range(1, ngen + 1):
        print(gen)
        # Select the next generation individuals
        #offspring_a = toolbox.select(pop, p_size)

        # Vary the pool of individuals
        offspring_a = varAnd(pop, toolbox, pc, pm)

        # Update pop a
        invalid_ind = [ind for ind in offspring_a if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the offspring
        pop[:] = toolbox.select(offspring_a + pop, p_size)

        # Append the current generation statistics to the logbook
        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    df = pd.DataFrame(logbook)

    return [toolbox.compile(expr=ind) for ind in pop], df, [str(ind) for ind in pop]


