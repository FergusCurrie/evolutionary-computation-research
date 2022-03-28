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

def single_predict(learner, x):
    '''
    does the unpacking here
    '''
    res = 1
    z = learner(*x) # unpack
    if z >= 0:
        res = 1
    else:
        res = 0
    return res


def array_predict(learner, X):
    vfunc = np.vectorize(learner)
    return vfunc(X)

def GP_predict(learner, X):
    """GP learner is simply a lambda. However it takes 5 arguments. 

    Args:
        learner (lambda): [description]
        X ([type]): [description]

    Returns:
        np.array  : (n_datapoints, )
    """
    assert(type(learner) != list)
    result = []
    for x in X:
        result.append(single_predict(learner,x))
    result = np.array(result)
    assert(result.shape[0] == X.shape[0])
    return np.array(result)