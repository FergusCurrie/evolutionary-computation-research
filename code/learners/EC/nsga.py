import random

from deap import creator, base, tools, algorithms


creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))
creator.create('Individual', list, fitness=creator.FitnessMulti)

def evalFitness(individual):
    return 0


toolbox = base.Toolbox()
toolbox.register('bit', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.bit, n=80)
toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=100)
toolbox.register('evaluate', evalFitness)
toolbox.register('mate', tools.cxUniform, indpb=0.1)
toolbox.register('mutate', toolbox.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selNSGA2)


pop = toolbox.population()
fits = toolbox.map(toolbox.evaluate, pop)
for fit , inds in zip(toolbox.evaluate, pop):
    inds.fitness.values = fit

for gen in range(50):
    offspring = algorithms.varOr(pop, toolbox, lambda_=100, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        inds.fitness.values = fit
    pop = toolbox.select(offspring + pop, k=100)



