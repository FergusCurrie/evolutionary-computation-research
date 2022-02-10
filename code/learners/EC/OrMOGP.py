

from distutils.log import error
from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
import operator
from code.learners.EC.deap_extra import get_pset, make_predictions
from code.metrics.classification_metrics import *
import pandas as pd 
import random


# https://github.com/DEAP/deap/blob/master/doc/api/tools.rst


def get_ypred(f, _X) -> np.array: # can abstract this from experiment_a gp predict
    result = []
    for x in _X:
        z = f(*x)
        if z >= 0:
            result.append(1)
        else:
            result.append(0)
    result = np.array(result)
    return result

def fitness_calculation(individual, toolbox, X, y, pop, pop_diversity):
    """
    Minimisation fitness calculation 

    Calcualte two objectives:
    1. Error Rate 
    2. Calculate diversity of individual as diversity(pop) - diversity(pop.remove(individual))
    Args:
        individual (deap gptree): individual to calculate fitness for
        toolbox (deap toolbox): toolbox deap uses for compiling individuals
        X (observations): dataset
        y (ground truth): dataset
        pop (list deap gptrees) : all individuals in population
        pop_diversity : diversity of pop 

    Returns:
        float :  fitness of this individual 
    """
    # First calculate obj1 
    func = toolbox.compile(expr=individual)
    err = error_rate(y, get_ypred(func, X))
    
    # Then get diversity from dict
    index = -1
    for i, ind in enumerate(pop):
        if ind == individual:
            index = i
    diveristy = pop_diversity[index]

    return err, diveristy,

def calculate_diversity_dict(individuals, toolbox, X, y) -> dict:
    """
    The job of this method is to calculate the diversity of a list of individuals. 

    This is a dirty solution, but we want the 'fitness function' to simple index from here.

    The dict will index from individual to diversity mesasure

    This has a O(MN) computation for averaging which presumably could be improved with 'cacheing' and numpy.
    Improving this method is a potential future research direction. 

    Args:
        individuals (deap gptree): each individual currently in ensemble
        toolbox (deap toolbox): toolbox deap uses for compiling individuals
        X (observations): dataset
        y (ground truth): dataset

    Returns:
        float :  diversity measure of this ensemble


    IM GOING TO PRESUME THAT THE EXPECTATION IS IMPLICIT
    """
    
    n = len(individuals) # number of datapoints 
    M = len(individuals) # number of models

    # Get the error rate per individual
    G = np.array([error_rate(y, get_ypred(toolbox.compile(expr=ind), X)) for ind in individuals]) 
    
    # Get the population mean
    g_ = np.mean(G)  

    # Get ommision means
    Gsj = ((g_ * n) - G ) / (M - 1)

    # Get diversity full 
    dfull = 1/(M-1) * np.sum(np.square(G - g_))

    # Get omission diversity
    Dommi = [1/(M-2) * np.sum(np.square(G - Gsj[j])) for j in range(M)]

    # Get diveristy meaasure (full - omission (500,))
    diversity = dfull - Dommi

    assert(diversity.shape[0] == len(individuals))

    # Create dictionry
    #dictionary = {key: value for (key, value) in zip(individuals, diversity)}

    return diversity
    

def gp_ormo_member_generation(X, y, params, seed):
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

    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * 2)  # min * n_objectives
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # Initalise tool box - NOTICE FOR OrMOGP we get the population from get_toolbox. Hacky and should be improved. 
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    pop = toolbox.population(n=p_size) # initialise the population to pass as final argument to fitnesss
    pop_diversity = calculate_diversity_dict(pop, toolbox, X, y)
    toolbox.register("evaluate", fitness_calculation, toolbox=toolbox, X=X, y=y, pop=pop, pop_diversity=pop_diversity)  # HERE?
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    

    # Stats
    mstats = tools.Statistics(lambda ind: ind.fitness.values)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"


    # Init pop fitness
    pop_diversity = calculate_diversity_dict(pop, toolbox, X, y)
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = mstats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)


    # Evolution process 
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring_a = toolbox.select(pop, len(pop))

        # Vary the pool of individuals
        offspring_a = varAnd(offspring_a, toolbox, pc, pm)
        
        # Update population diversity
        pop_diversity = calculate_diversity_dict(pop, toolbox, X, y)

        # Update pop a
        invalid_ind = [ind for ind in offspring_a if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the offspring
        pop[:] = offspring_a

        # Append the current generation statistics to the logbook
        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    df = pd.DataFrame(logbook)

    return [toolbox.compile(expr=ind) for ind in pop], df
