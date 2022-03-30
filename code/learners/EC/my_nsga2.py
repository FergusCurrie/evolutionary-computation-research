
"""

Code copied from deap. Reimplementing crowding dtiance for implementing:
    - pairwise failure crediting
    - negative correlation learning

Clearly diveristy is the second objective here. 


may be able to shave a bit of time about by dropping the dimension of left down to the front 

"""
import bisect
from collections import defaultdict, namedtuple
from distutils.log import error
from inspect import stack
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random

import numpy as np
from pyrsistent import v
from code.decision_fusion.voting import binary_voting

from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import error_rate
import time


def selncl(individuals, k, X, y, toolbox, nd='standard', div='crowding'):
    """Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if 0: # these variables are coming through fine 
        print(k)
        print(len(individuals))
        print(X)
        assert(False)
    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))
    
    if div == 'crowding': # standard nsga2 diveristy
        for front in pareto_fronts:
            assignCrowdingDist(front)
    if div == 'ncl': # ncl diversity
        assignNCLDist(individuals, None, X, y, toolbox)
    if div == 'pf': # pairwise failure crediting 
        assignPFDist(individuals, None, X, y, toolbox)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


def sortNondominated(individuals, k, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

def assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        
        crowd.sort(key=lambda element: element[0][i]) # for individ incrowd, fitness of ind, i # obj
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist
    



def assignNCLDist(ensemble, individuals, X,y, toolbox):
    """
    ensemble is all the gp trees
    individuals is the front 

    Copy of assignCrowdingDistance, but calculates the distance using negative correlation learning 

    LOWER NCL IS BETTER DIVERSITY, THIS MEANS WE SHOULD RETURN A NEGATIVE AS THIS SHOULD FUNCTION LIKE CROWDING DISTANCE
    WHICH PENALISES LOWER DISTANCE VALUES. 

    Individuals are already a complied lambda

    Despite technically occuring at front level we need to calculate ensemble level predicitons 

    Currently using a loopy implementation which may serve to be too slow 

    We only calculate NCL for the front, but compare to the wider ensemble 
    """

    if len(ensemble) == 0:
        return
    num_classes = len(np.unique(y)) # binary - can be changed latter 
    assert(num_classes == 2)

    # first calculate the matrices for fast calculation on each class . will be shape ensemble x class x class
    outers = []
    for k in range(num_classes):
        mask = y == k 
        g = np.zeros((len(ensemble), len(y[mask]))) # shape of learners x observations  
        for j,e in enumerate(ensemble):
            g[j, :] = GP_predict(toolbox.compile(expr=e), X[mask]) # can we pass single datpoints? 
        G = (1 / 1 + np.exp(g)) # elementwise.  # only giving euler number 
        E = binary_voting(g) # (len())
        outer = G - E
        outers.append(outer)


    
    # loop calculatin of ncl for each member 
    ncls = np.zeros(len(ensemble))
    for k in range(num_classes):
        mask = y == k # mask where we look at correct class 
        Nc = len(y[mask])
        M = len(ensemble) # number non-dominated solutins? 
        outer = outers[k] # get correct outer 

        rhs = np.copy(outer)
        np.fill_diagonal(rhs, 0)
        np.sum(rhs, axis=0) # sum along model
        final = outer * rhs
        final = np.sum(final, axis=1) # sum along class
        final = final * (1/(M*Nc))
        ncls += final
    ncls = ncls * -0.5


    # set into individuals as would work for crowding distance 
    # set negative ncl as crowding distance is optimally high, ncl optimally low 
    for i, dist in enumerate(ncls):
        ensemble[i].fitness.crowding_dist = dist
    


def assignPFDist(ensemble, individuals, X,y, toolbox, weighting=0.5):
    """
    Copy of assignCrowdingDistance, but calculates the distance using pairwise failure crediting 
    """
    if len(ensemble) == 0:
        return

    # Set distances to 0
    for e in ensemble:
        e.fitness.crowding_dist = 0

    M = len(ensemble)
    for k in range(len(np.unique(y))): # for each class
        mask = y == k 
        Nc = len(y[mask])
        g = np.zeros((len(ensemble), len(y[mask]))) # shape of learners x observations  
        er = np.zeros(M,)
        for j,e in enumerate(ensemble):
            g[j, :] = GP_predict(toolbox.compile(expr=e), X[mask]) # can we pass single datpoints? 
            er[j] = error_rate(g[j, :], y[mask])
        er_outer = np.subtract.outer(er, -er)

        z = np.ones((Nc, M, M))
        for o in range(Nc):
            for q in range(M):
                z[o,q,:] = g[q,o].reshape(1,) - g[:,o]

        # Wierd chunk of code but basically 0s and 2s are agreement. 1s are disagreement
        # Want to update for 0s agreement and 1s disagreement
        z[z == 2] = 1

        #print(z.shape)
        z = np.sum(z, axis=0) # sum down Nc axis
        assert(z.shape == (M,M))

        # Divide oute pairwise model error rate
        z = z / er_outer

        # ignore where they are the same 
        np.fill_diagonal(z, 0)

        # sum along either axis
        z = np.sum(z, axis=0)
        assert(z.shape == (M,))

        # scale by T (population size = M)
        z =  (1 / (M - 1)) * z

        # update 
        for i, dist in enumerate(z):
            if k == 0:
                ensemble[i].fitness.crowding_dist -= (dist * weighting) # minus to comply with other nsga2 code 
            else:
                ensemble[i].fitness.crowding_dist -= (dist * (1-weighting))
