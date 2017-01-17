
import get_neab

"""
Code from the deap framework, available at:
https://code.google.com/p/deap/source/browse/examples/ga/onemax_short.py
Conversion to its parallel form took two lines:
from scoop import futures
toolbox.register("map", futures.map)
"""
import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0,-1.0,-1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

class Individual(list):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error=None
        self.sciunitscore=[]
        self.model=None
        self.error=None
        self.results=None
        self.name=''
        self.attrs={}
        self.params=None
        self.score=None

toolbox = base.Toolbox()

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 4

param=['vr','a','b']
rov=[]
#rov0 = np.linspace(-65,-55,2)
#rov1 = np.linspace(0.015,0.045,2)
#rov2 = np.linspace(-0.0010,-0.0035,2)

rov0 = np.linspace(-65,-55,1000)
rov1 = np.linspace(0.015,0.045,1000)
rov2 = np.linspace(-0.0010,-0.0035,1000)
rov.append(rov0)
rov.append(rov1)
rov.append(rov2)
seed_in=1

BOUND_LOW=[ np.min(i) for i in rov ]
BOUND_UP=[ np.max(i) for i in rov ]
NDIM = len(rov)
LOCAL_RESULTS=[]

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
model = ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()
model.local_run()


def func2map(ind):

    for i, p in enumerate(param):
        name_value=str(ind[i])
        #reformate values.
        model.name=name_value
        if i==0:
            attrs={'//izhikevich2007Cell':{p:name_value }}
        else:
            attrs['//izhikevich2007Cell'][p]=name_value

    ind.attrs=attrs

    model.update_run_params(attrs)

    ind.params=[]
    for i in attrs['//izhikevich2007Cell'].values():
        if hasattr(ind,'params'):
            ind.params.append(i)

    ind.results=model.results
    score = suite.judge(model)
    try:
        ind.error = score.sort_keys()
    except:
        print(dir(score))
        ind.error = score.sort_keys.values[0]
        print(type(score))
        print(score)

    return ind

def evaluate(individual):#This method must be pickle-able for scoop to work.
    individual=func2map(individual)
    error=individual.error
    assert individual.results
    #LOCAL_RESULTS.append(individual.results)

    return error[0],error[1],error[2],error[3],error[4],



toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)

def plotss(pop,gen):
    import matplotlib.pyplot as plt
    #print(utils.getHosts())
    if hasattr(pop[0],'results'):
        plt.hold(True)

        for ind in pop:
            if hasattr(ind,'results'):
                plt.plot(ind.results['t'],ind.results['vm'])
                plt.xlabel(str(ind.attrs))
                #str(scoop.worker)+
        plt.savefig('snap_shot_at '+str(gen)+str(utils.getHosts())+'.png')
        plt.hold(False)
        plt.clf()
    #return 0

def main(seed=None):

    random.seed(seed)

    NGEN=4
    MU=12

    CXPB = 0.9
    import numpy as numpy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        print(gen)
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        #assert ind.results

        offspring = [toolbox.clone(ind) for ind in offspring]
        #print('cloning not true clone')


        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            #del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            # Select the next generation population
        #pop = toolbox.select(pop + offspring, MU)
        pop = toolbox.select(offspring, MU)
        plotss(pop,gen)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)


    return pop, logbook

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    start_time=time.time()

    pop, stats = main()
    finish_time=time.time()
    ga_time=finish_time-start_time
    pop.sort(key=lambda x: x.fitness.values)
    import numpy
    front = numpy.array([ind.fitness.values for ind in pop])
    plt.scatter(front[:,0], front[:,1], front[:,2], front[:,3])
    plt.axis("tight")
    plt.savefig('front.png')

    print(stats)


    # plt.show()
