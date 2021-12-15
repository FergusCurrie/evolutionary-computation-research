from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
from deap import algorithms
import operator


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

def calculate_confusion_matrix(y_true, y_pred):
    confusion_matrix = [[0,0],[0,0]] # first index is 1=true/0=false, second is 1=positive, 0=negative
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            if yp == 1:
                confusion_matrix[0][1] += 1
            if yp == 0:
                confusion_matrix[1][0] += 1
        elif yt == 1:
            if yp == 1:
                confusion_matrix[1][1] += 1
            elif yp == 0:
                confusion_matrix[0][0] += 1
    return confusion_matrix

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


def get_stats():

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)
    return mstats

def member_evaluation(individual, toolbox, X, y):
    func = toolbox.compile(expr=individual)
    ypred = make_predictions(X, func)
    confusion_matrix = calculate_confusion_matrix(y, ypred)
    acc = accuracy(confusion_matrix)
    return acc


def member_selection(population, toolbox, X, y):
    # First sort population 
    sorted_pop = sorted(population, key=lambda e : member_evaluation(individual=e, toolbox=toolbox,X=X,y=y), reverse=True) # DESCENDING 

    # Now loop through an 
    L = []
    best = [-1, -1] # ind, acc
    for i in range(len(sorted_pop)-1):
        L = sorted_pop[0:i+1]
        F = [toolbox.compile(expr=individual) for individual in L]
        ypred = []
        for x in X:
            _yp = 0
            preds = np.array([single_pred(x, f) for f in F])
            if len(preds[preds == 0]) < len(preds[preds == 1]):
                _yp = 1
            ypred.append(_yp)
        confusion_matrix = calculate_confusion_matrix(y, ypred)
        acc = accuracy(confusion_matrix)
        if best[1] < acc:
            best = [i, acc]
    return population[0:best[0]+1]



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

    #species = [toolbox.species() for _ in range(2)]

    # Initalise stats
    mstats = get_stats()

    # Run GP
    pop = toolbox.population(n=p_size)


    # Init pop fitness
    # Update fitness in population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    # Evolution process 
    for gen in range(1, ngen + 1):
        if verbose:
            print(f'gen={gen}')
        # Select the next generation individuals
        offspring_a = toolbox.select(pop, p_size)

        # Vary the pool of individuals
        offspring_a = varAnd(offspring_a, toolbox, pc, pm)
        #offspring_a = algorithms.varOr(pop, toolbox, cxpb=pc, mutpb=pm)

        # Update fitness in population
        invalid_ind = [ind for ind in offspring_a if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #[print(ind.fitness.valid) for ind in offspring_a]

        # Replace the current population by the offspring
        #pop[:] = offspring_a + pop
        pop = toolbox.select(offspring_a + pop, k=p_size) # something to do with having 200 here. 

    ensemble = member_selection(pop, toolbox, X, y)
    return [toolbox.compile(expr=ind) for ind in ensemble]

    # run over 


    # Compile best function, apply to test and print
    #funca = toolbox.compile(expr=CV[0])


