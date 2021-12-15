# Code for MOGP 


from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
from deap import algorithms
import math
import operator
import pandas as pd 
import random

from code.metrics.classification_metrics import *


def my_if(a, b, c):
    if a > 0:
        return b
    return c

def protectedDiv(left, right):
    if right <= 0:
        return 9999999 # val?
    try:
        return left / right
    except ZeroDivisionError:
        return 9999999 # val

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

def get_toolbox(pset, t_size, max_depth, X, y):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_calculation, toolbox=toolbox, X=X, y=y) # HERE?
    toolbox.register("select", tools.selTournament, tournsize=t_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    return toolbox

def make_predictions(_X, f):
    ypred = []
    for x in _X:
        yp = 1
        if f(*x) < 0:
            yp = 0
        ypred.append(yp)
    return np.array(ypred)

def fitness_calculation(individual, toolbox, X, y, w=0.5):
    """
    Fitness function. Compiles tree then calls mse function
    """
    func = toolbox.compile(expr=individual)
    # Calculated the 'ave' function
    ypred = make_predictions(X, func)
    confusion_matrix = calculate_confusion_matrix(y, ypred)
    return ave(confusion_matrix, w),



def get_stats():

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)
    return mstats

def gp_member_generation(X,y, p_size, max_depth, pc, pm, ngen, t_size,verbose=False):


    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # max
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Initalise tool box
    toolbox = get_toolbox(pset, t_size, max_depth, X, y)

    # Initalise stats
    mstats = get_stats()

    # Run GP
    pop = toolbox.population(n=p_size)

    halloffame = tools.HallOfFame(1)

    # Evolution process 
    for gen in range(1, ngen + 1):
        print(f'gen={gen}')
        # Select the next generation individuals
        offspring_a = toolbox.select(pop, len(pop))

        # Vary the pool of individuals
        offspring_a = varAnd(offspring_a, toolbox, pc, pm)

        # Update pop a
        invalid_ind = [ind for ind in offspring_a if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring_a)


        # Replace the current population by the offspring
        pop[:] = offspring_a

    return [toolbox.compile(ind) for ind in pop]

    # Compile best function, apply to test and print
    
    # funca = toolboxa.compile(expr=CV[0])


