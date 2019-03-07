
def active_values(keyed):
    DURATION = 1000.0*pq.ms
    DELAY = 100.0*pq.ms
    keyed['injected_square_current'] = {}

    keyed['injected_square_current']['delay']= DELAY
    keyed['injected_square_current']['duration'] = DURATION
    keyed['injected_square_current']['amplitude'] = dtc.rheobase
    return keyed
def passive_values(keyed):
    DURATION = 500.0*pq.ms
    DELAY = 200.0*pq.ms
    keyed['injected_square_current'] = {}
    keyed['injected_square_current']['delay']= DELAY
    keyed['injected_square_current']['duration'] = DURATION
    keyed['injected_square_current']['amplitude'] = -10*pq.pA
    return keyed

def format_test(dtc):
    '''
    pre format the current injection dictionary based on pre computed
    rheobase values of current injection.
    This is much like the hooked method from the old get neab file.
    '''
    dtc.vtest = {}
    dtc.tests = switch_logic(dtc.tests)

    for k,v in enumerate(dtc.tests):
        dtc.vtest[k] = {}
        dtc.vtest[k]['injected_square_current'] = {}

        if v.active == True and v.passive == False:
            #keyed = dtc.vtest[k]
            #dtc.vtest[k] = active_values(keyed)

            DURATION = 1000.0*pq.ms
            DELAY = 100.0*pq.ms
            dtc.vtest[k]['injected_square_current']['delay']= DELAY
            dtc.vtest[k]['injected_square_current']['duration'] = DURATION
            dtc.vtest[k]['injected_square_current']['amplitude'] = dtc.rheobase
            #'''
        if v.passive == True and v.active == False:
            #keyed = dtc.vtest[k]
            #dtc.vtest[k] = passive_values(keyed)
            #'''
            DURATION = 500.0*pq.ms
            DELAY = 200.0*pq.ms
            dtc.vtest[k]['injected_square_current'] = {}
            dtc.vtest[k]['injected_square_current']['delay']= DELAY
            dtc.vtest[k]['injected_square_current']['duration'] = DURATION
            dtc.vtest[k]['injected_square_current']['amplitude'] = -10*pq.pA
            #'''
        # not returned so actually not effective
        #v.params = dtc.vest[k]

    return dtc

from neuronunit.optimzation import optimization_management

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#toolbox.register("evaluate", benchmarks.rastrigin)
self.toolbox.register("evaluate", optimization_management.evaluate)


import matplotlib.pyplot as plt

# Problem size
N = 10
NGEN = 125
#Then, it does not get any harder. Once a Strategy is instantiated, its generate() and update() methods are registered in the toolbox for uses in the eaGenerateUpdate() algorithm. The generate() method is set to produce the created Individual class. The random number generator from numpy is seeded because the cma module draws all its number from it.
#https://github.com/DEAP/deap/blob/master/examples/es/cma_plotting.py
pop = self.set_pop()
self.toolbox.register("evaluate", optimization_management.evaluate)

strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=20*N)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

for gen in range(NGEN):
    # Generate a new population
    population = toolbox.generate()
    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Update the strategy with the evaluated individuals
    toolbox.update(population)

    # Update the hall of fame and the statistics with the
    # currently evaluated population
    halloffame.update(population)
    record = stats.compile(population)
    logbook.record(evals=len(population), gen=gen, **record)

    if verbose:
        print(logbook.stream)

    # Save more data along the evolution for latter plotting
    # diagD is sorted and sqrooted in the update method
    sigma[gen] = strategy.sigma
    axis_ratio[gen] = max(strategy.diagD)**2/min(strategy.diagD)**2
    diagD[gen, :N] = strategy.diagD**2
    fbest[gen] = halloffame[0].fitness.values
    best[gen, :N] = halloffame[0]
    std[gen, :N] = numpy.std(population, axis=0)


fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Update the strategy with the evaluated individuals
toolbox.update(population)

# Update the hall of fame and the statistics with the
# currently evaluated population
halloffame.update(population)
record = stats.compile(population)
logbook.record(evals=len(population), gen=gen, **record)

if verbose:
    print(logbook.stream)

# Save more data along the evolution for latter plotting
# diagD is sorted and sqrooted in the update method
sigma[gen] = strategy.sigma
axis_ratio[gen] = max(strategy.diagD)**2/min(strategy.diagD)**2
diagD[gen, :N] = strategy.diagD**2
fbest[gen] = halloffame[0].fitness.values
best[gen, :N] = halloffame[0]
std[gen, :N] = numpy.std(population, axis=0)
