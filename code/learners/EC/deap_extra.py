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

from code.decision_fusion.voting import majority_voting
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
    pset.addPrimitive(operator.add, 2) # second number is arity which means how mnay nodes leadi nto it 
    pset.addPrimitive(operator.sub, 2, name="sub")
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

def thresholding(z, x, unique_classes):
    radius = 0.5
    res = -1
    n_classes = len(unique_classes)
    lowest_threshold = 0.3 * n_classes * -1
    for i_class in range(n_classes):
        if i_class == n_classes-1: # if we get to the limit on the positive side then we classify to infity here
            res = i_class
            break
        if z < lowest_threshold + (i_class * (radius * 2)):
            res = i_class
            break

    assert(res != -1)
    return res

def single_predict(learner, x,unique_classes):
    '''
    does the unpacking here
    '''
    z = learner(*x) # unpack
    return z


def array_predict(learner, X):
    vfunc = np.vectorize(learner)
    return vfunc(X)



def GP_predict(learner, X, n_classes):
    """GP learner is simply a lambda. However it takes 5 arguments. 

    Args:
        learner (lambda): [description]
        X ([type]): [description]
        n : num thresholds 

    Returns:
        np.array  : (n_datapoints, )
    """
    assert(type(learner) != list)
    result = []
    for x in X:
        p = single_predict(learner,x, n_classes)
        result.append(thresholding(p,x, n_classes))

    result = np.array(result)
    assert(result.shape[0] == X.shape[0])
    return np.array(result)


def raw_bag_GP_predict(bag, X, n_classes):
    """Same as above but instead the learner is a list of trees 

    Args:
        learner (_type_): _description_
        X (_type_): (data, feat)
        n_classes (_type_): _description_
    """
    dv = []
    for tree in bag:
        temp = []
        for x in X:
            p = single_predict(tree, x, n_classes)
            temp.append(p)
        dv.append(temp)
    dv = np.array(dv) #  (1, data)
    outer_vote = np.sum(dv, axis=0)
    return outer_vote # (databpoitns )