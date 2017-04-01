
import os,sys
import numpy as np
import matplotlib.pyplot as plt
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

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scoop import futures

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

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

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)




def func2map(ind):

    ind.sciunitscore={}
    #model=model.load_model()
    for i, p in enumerate(param):
        name_value=str(ind[i])
        #reformate values.
        model.name=name_value
        if i==0:
            attrs={'//izhikevich2007Cell':{p:name_value }}
        else:
            attrs['//izhikevich2007Cell'][p]=name_value

    model.update_run_params(attrs)
    import copy

    ind.results=copy.copy(model.results)

    score = suite.judge(model)
    ind.error = copy.copy(score.sort_keys.values[0])
    return ind

def evaluate(individual):#This method must be pickle-able for scoop to work.
    #print('hello from before error')
    func2map(individual)
    error=individual.error
    return error[0],error[1],error[2],error[3],error[4],

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("map", futures.map)

def main():
    random.seed(64)

    pop = toolbox.population(n=3)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    main()
