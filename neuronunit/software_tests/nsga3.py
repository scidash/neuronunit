
import os,sys
import numpy as np

import matplotlib as matplotlib
matplotlib.use('agg')
import quantities as pq
import sciunit

#Over ride any neuron units in the PYTHON_PATH with this one.
#only appropriate for development.
thisnu = str(os.getcwd())+'/../..'
sys.path.insert(0,thisnu)
print(sys.path)

import neuronunit
from neuronunit import aibs
from neuronunit.models.reduced import ReducedModel
import pdb
import pickle
from scoop import futures

IZHIKEVICH_PATH = os.getcwd()+str('/NeuroML2') # Replace this the path to your
LEMS_MODEL_PATH = IZHIKEVICH_PATH+str('/LEMS_2007One.xml')


import time

from pyneuroml import pynml

import quantities as pq
from neuronunit import tests as nu_tests, neuroelectro
neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []

dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                        # Primary visual area, layer 5 neuron.
observation = aibs.get_observation(dataset_id,'rheobase')


if os.path.exists(str(os.getcwd())+"/neuroelectro.pickle"):
    print('attempting to recover from pickled file')
    with open('neuroelectro.pickle', 'rb') as handle:
        tests = pickle.load(handle)

else:
    print('checked path:')
    print(str(os.getcwd())+"/neuroelectro.pickle")
    print('no pickled file found. Commencing time intensive Download')

    #(nu_tests.TimeConstantTest,None),                           (nu_tests.InjectedCurrentAPAmplitudeTest,None),
    tests += [nu_tests.RheobaseTest(observation=observation)]
    test_class_params = [(nu_tests.InputResistanceTest,None),
                         (nu_tests.RestingPotentialTest,None),
                         (nu_tests.InjectedCurrentAPWidthTest,None),
                         (nu_tests.InjectedCurrentAPThresholdTest,None)]



    for cls,params in test_class_params:
        #use of the variable 'neuron' in this conext conflicts with the module name 'neuron'
        #at the moment it doesn't seem to matter as neuron is encapsulated in a class, but this could cause problems in the future.


        observation = cls.neuroelectro_summary_observation(neuron)
        tests += [cls(observation,params=params)]

    with open('neuroelectro.pickle', 'wb') as handle:
        pickle.dump(tests, handle)

def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']#first find a value for rheobase
    #then proceed with other optimizing other parameters.


    print(len(tests))
    #pdb.set_trace()
    for i in [2,3,4]:
        # Set current injection to just suprathreshold
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01


#Do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
#Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.

hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests,hooks=hooks)


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
rov0=[-65,-55]
rov1=[0.015,0.045]
rov2=[-0.0010,-0.0035]
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
    ind.params=[]
    for i in attrs['//izhikevich2007Cell'].values():
        ind.params.append(i)
    ind.attrs=attrs
    model.update_run_params(attrs)


    ind.results=model.results
    score = suite.judge(model)
    ind.error = score.sort_keys.values[0]
    return ind

def evaluate(individual):#This method must be pickle-able for scoop to work.
    individual=func2map(individual)
    error=individual.error
    assert individual.results
    LOCAL_RESULTS.append(individual.results)

    return error[0],error[1],error[2],error[3],error[4],



toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)



def main(seed=None):
    start_time=time.time()

    random.seed(seed)

    NGEN=3
    MU=8

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

    if hasattr(pop[0],'results'):
        import matplotlib.pyplot as plt
        plt.hold(True)

        for ind in pop:
            if hasattr(ind,'results'):
                plt.plot(ind.results['t'],ind.results['vm'])
        plt.savefig('initial_pop.png')

        plt.hold(False)
        plt.clf()
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
        #assert ind.results

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)

        if hasattr(pop[0],'results'):
            import matplotlib.pyplot as plt
            plt.hold(True)

            for ind in pop:
                if hasattr(ind,'results'):
                    plt.plot(ind.results['t'],ind.results['vm'])
                    plt.xlabel(str(ind.attrs))
            plt.savefig('snap_shot_at '+str(gen)+'.png')

            plt.hold(False)
            plt.clf()

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)


    #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    finish_time=time.time()
    ga_time=finish_time-start_time

    import matplotlib.pyplot as plt

    print(len(pop))
    plt.hold(True)
    plt.title('time expended '+str(ga_time/60.0)+' minutes, ngen*pop_size: '+str(NGEN*MU)+'.png')

    for i,ind in enumerate(pop):
        if hasattr(ind,'results'):
            plt.plot(ind.results['t'],ind.results['vm'])
        if i==0 and hasattr(ind,'attrs'):
            plt.xlabel(str(ind.attrs))
    plt.savefig('evolved_pop.png')
    plt.hold(False)
    plt.clf()
    plt.hold(True)
    plt.title('time expended '+str(ga_time/60.0)+' minutes, ngen*pop_size: '+str(NGEN*MU)+'.png')

    for i,ind in enumerate(pop):
        if(i<5):
            if hasattr(ind,'results'):
                plt.plot(ind.results['t'],ind.results['vm'])
        if i==0 and hasattr(ind,'attrs'):
            plt.xlabel(str(ind.attrs))
    plt.savefig('evolved_pop_5.png')
    plt.hold(False)

    plt.clf()


    pop.sort(key=lambda x: x.fitness.values)
    #print("Convergence: ", convergence(pop, optimal_front))
    #print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    import numpy
    front = numpy.array([ind.fitness.values for ind in pop])
    front_params = numpy.array([ind.params for ind in pop])

    #optimal_front = numpy.array(optimal_front)
    #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], front[:,2], front[:,3])
    plt.axis("tight")
    plt.savefig('front.png')

    plt.clf()
    plt.scatter(front_params[:,0], front_params[:,1], front_params[:,2], front_params[:,3])
    plt.axis("tight")
    plt.savefig('front_params.png')

    return pop, logbook

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    pop, stats = main()

    pop.sort(key=lambda x: x.fitness.values)

    print(stats)

    # plt.show()
