from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
from deap import algorithms
import operator
from code.learners.EC.deap_extra import my_if, protectedDiv, get_pset, get_stats, update_fitness
from code.metrics.classification_metrics import *
from code.member_selection.offEEL import offEEL

def get_toolbox(pset, max_depth, X, y):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_calculation, toolbox=toolbox, X=X, y=y) # HERE?
    toolbox.register('select', tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    return toolbox


def single_pred(x, f):
    yp = 1
    if f(*x) < 0:
        yp = 0
    return yp

def make_predictions(_X, f):
    return np.array([single_pred(x, f) for x in _X])

def accuracy(cm):
    """confusion matrix # first index is 1=true/0=false, second is 1=positive, 0=negative """
    return (cm[1][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1])

def fitness_calculation(individual, toolbox, X, y):
    """
    Fitness function. Compiles tree then calls mse function
    """
    func = toolbox.compile(expr=individual)

    # Calculate accuracy on class 0 
    ypred = make_predictions(X[y == 0], func)
    confusion_matrix = calculate_confusion_matrix(y[y == 0], np.array(ypred))
    class_0_acc = accuracy(confusion_matrix)

    # Calculate accuracy on class 1
    ypred = make_predictions(X[y == 1], func)
    confusion_matrix = calculate_confusion_matrix(y[y == 1], np.array(ypred))
    class_1_acc = accuracy(confusion_matrix)
    return class_0_acc, class_1_acc,

def member_evaluation(individual, toolbox, X, y):
    func = toolbox.compile(expr=individual)
    ypred = make_predictions(X, func)
    confusion_matrix = calculate_confusion_matrix(y, ypred)
    acc = accuracy(confusion_matrix)
    return acc



def gp_mo_member_generation(X,y, p_size, max_depth, pc, pm, ngen, verbose=False):
    """Generation an ensemble of GP trees using multi-objective tournament selection. 

    Args:
        X (np.array): training data 
        y (np.array): training labels 
        p_size (int): number of individuals in population
        max_depth (int): max depth of gp tree. limits program size
        pc (float): probability of individual crossing
        pm (float): probability of individual mutating
        ngen (int): number of generations before termination
        verbose (bool, optional): whether to print progress of training or not 

    Returns:
        [list]: ensemble members... which are just lambda functions 
    """
    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0)) # min
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # Initalise tool box
    toolbox = get_toolbox(pset, max_depth, X, y)

    # Run GP
    pop = toolbox.population(n=p_size)

    # Init pop fitness
    update_fitness(pop, toolbox)

    # Evolution process 
    for gen in range(1, ngen + 1):
        if verbose:
            print(f'gen={gen}')

        # Select the next generation individuals
        offspring_a = toolbox.select(pop, p_size)

        # Vary the pool of individuals
        offspring_a = varAnd(offspring_a, toolbox, pc, pm)

        # Update fitness in population
        update_fitness(pop, toolbox)

        # Replace the current population by the offspring
        pop = toolbox.select(offspring_a + pop, k=p_size) # something to do with having 200 here. 

    ensemble = offEEL(pop, toolbox, X, y)
    return [toolbox.compile(expr=ind) for ind in ensemble]