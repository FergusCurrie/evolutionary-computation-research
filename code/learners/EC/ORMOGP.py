

from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
import operator
from code.learners.EC.deap_extra import GP_predict, get_pset, single_predict, array_predict
from code.metrics.classification_metrics import accuracy
import pandas as pd 
import random
import time 

def fitness_calculation(individual, toolbox, X, y, pop, pop_diversity, obj1):
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
    # First calculate obj1 - accuracy
    func = toolbox.compile(expr=individual)
    err = obj1(y, GP_predict(func, X))
    
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

    Order of expectation matters, we must place variance inside expectation. 

    Args:
        individuals (deap gptree): each individual currently in ensemble
        toolbox (deap toolbox): toolbox deap uses for compiling individuals
        X (observations): dataset
        y (ground truth): dataset

    Returns:
        float :  diversity measure of this ensemble
    """
    
    M = len(individuals) # number of models
    N = len(X) # number of observations 

    In = np.tile(X, (M,1,1)) # i,j is curently ith data point
    P = np.zeros((M,N))

    for j in range(M):
        ind = individuals[j]
        f = toolbox.compile(expr=ind)
        func = lambda x : single_predict(f, x)
        P[j] = np.apply_along_axis(func, axis=1, arr=In[j,:,:]) # apply to a 1d slice - 1d slice being the features 
    assert(P.shape == (M,N))

    # Calculate loss.. # P is essemtially 0/1 loss. - so 0 is good 
    G = (P == np.tile(y, (M,1))) # i,j is curently ith data point  # ()
    assert(G.shape == (M,N))
    
    g_ = np.mean(G, axis=0) # mean along all models  - mean per
    assert(g_.shape == (N,))

    # Get ommision means - im happy with this 
    Gsj_ = ((g_ * M) - G ) / (M - 1) # (n x m) - doing for every observation

    assert(Gsj_.shape == ((M,N)))

    # Get diveristy full
    diversity_full = np.var(G, axis=0)               # Variance
    diversity_full = np.mean(diversity_full, axis=0) # Expectation
    
    # Get diversity per omission - the reshaping is to prevent an elementwise subtraction
    Q = G.reshape(M,N,1) - Gsj_.reshape(N,M) 
    assert(Q.shape == (M,N,M))
    
    # Zero the diagonal where model dimensions overlap
    for i in range(M):
        Q[i,:,i] = 0

    # First the variance along the model ommisions axis - axis 0 is omission
    diversity_omissions = np.sum(np.square(Q), axis=0) / (M - 1) # Have zeroed once per this dim so dont count to avg
    assert(diversity_omissions.shape == (N,M))

    # Then the expectation along the observations axis
    diversity_omissions = np.mean(diversity_omissions, axis=0) 
    assert(diversity_omissions.shape == (M,))

    # Final diversity - remaining model axis 
    #diversity = diversity_full - diversity_omissions
    diversity = -diversity_omissions
    assert(diversity.shape == (M,))

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
    obj1 = params['obj1'] # what function to use for obj1

    # Initalise primitives
    pset = get_pset(num_args=X.shape[1])

    # Initialise GP settings
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * 2)  # min * n_objectives
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # Initalise tool box - NOTICE FOR OrMOGP we get the population from get_toolbox. Hacky and should be improved. 
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    pop = toolbox.population(n=p_size) # initialise the population to pass as final argument to fitnesss
    pop_diversity = calculate_diversity_dict(pop, toolbox, X, y)
    #return pop_diversity, [toolbox.compile(expr=ind) for ind in pop]
    toolbox.register("evaluate", fitness_calculation, toolbox=toolbox, X=X, y=y, pop=pop, pop_diversity=pop_diversity, obj1=obj1)  # HERE?
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

    return [toolbox.compile(expr=ind) for ind in pop], df, [str(ind) for ind in pop]




def test_div_helper(X, y, params, fs):
    return

    # Now we want to be able to confirm this
    G = 0
    e = 0 
    gsmeans = []
    for X_, y_ in zip(X, y):
        gs = []# all predictions through loss for this observation
        for f in fs:
            p = single_predict(f, X_)
            if p == y_:
                g = 1
            else:
                g = 0
            G += g# just for total comparison
            # loss applied, now variance?
            gs.append(g)
        gs_mean = np.mean(np.array(gs))
        gsmeans.append(gs_mean)
        s = 0 # Sum
        for g in gs:
            s += (np.square(g - gs_mean)) 
        e += s / (len(fs) + 1)
    return e / len(X), 
        


def test(X, y, params):
    """

    Making sure we havent completlyt fucked how this should work for diversity calc

    """
    return
    
    pred_d, fs = gp_ormo_member_generation(X, y, params, 12)
    print(pred_d)

    d = test_div_helper(X, y, params, fs)
    # print(d)


    true_d = []
    for i,f in enumerate(fs):
        print(i)
        t = [ff for ff in fs if ff != f]
        assert(len(t)+1 == len(fs))
        d = test_div_helper(X, y, params, t)
        true_d.append(d)
    true_d = np.array(true_d)

    print(true_d)

    
