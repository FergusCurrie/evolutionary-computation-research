"""
Methods common to any DEAP gp, just for decluttering. 
"""

from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import math
import operator 
import numpy as np
import warnings
warnings.simplefilter("ignore")

def update_fitness(pop, toolbox):
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

def my_if(a, b, c):
    if a > 0:
        return b
    return c

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 10

def get_pset(num_args):
    pset = gp.PrimitiveSet("MAIN", num_args)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(my_if, 3)
    for n in range(num_args):
        pset.renameArguments(ARG0=f'x{n}')
    pset.addTerminal(3)
    pset.addTerminal(2)
    pset.addTerminal(1)
    return pset

def get_stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)
    return mstats


def make_predictions(_X, f):
    """Uses a function to make a thresholded prediction across a given 2d dataset. For binary data 

    Args:
        _X (np.array): dataset
        f (callable): function

    Returns:
        np.array: array of predictions 
    """
    ypred = []
    for x in _X:
        yp = 1
        if f(*x) < 0:
            yp = 0
        ypred.append(yp)
    return np.array(ypred)