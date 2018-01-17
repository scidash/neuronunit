##
# This module has been depreciated.
# Any functional contents, that it has are being moved into 
# nsga_parallel and BluePyOpt/deapExt/algorithms
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##



import os
from neuronunit.models import backends
import neuronunit
print(neuronunit.models.__file__)
from neuronunit.models.reduced import ReducedModel
from ipyparallel import require
import ipyparallel as ipp

rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]
'''
rc.Client.become_dask()
'''
class Individual(object):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error = None
        self.results = None
        self.name=''
        self.attrs = {}
        self.params = None
        self.score = None
        self.fitness = None
        self.lookup = {}
        self.rheobase = None
        self.fitness = creator.FitnessMin

@require('numpy, deap','random')
def import_list(ipp,subset,numb_err_f,NDIM):
    #ipp = ipp
    #import ipyparallel as ipp
    #rc = ipp.Client(profile='default')
    #dview = rc[:]
    Individual = ipp.Reference('Individual')
    from deap import base, creator, tools
    import deap
    import random
    history = deap.tools.History()
    toolbox = base.Toolbox()
    import numpy as np
    ##
    # Range of the genes:
    ##
    BOUND_LOW = [ np.min(i) for i in subset.values() ]
    BOUND_UP = [ np.max(i) for i in subset.values() ]
    ##
    # number of objectives/error functions
    ##
    #NDIM = numb_err_f
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    # weights vector should compliment a numpy matrix of eigenvalues and other values
    #creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    weights = tuple([1.0 for i in range(0,numb_err_f)])
    #weights = tuple([-1.0 for i in range(0,NDIM)])
    creator.create("FitnessMin", base.Fitness, weights = weights)
    creator.create("Individual", list, fitness = creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, len(BOUND_UP))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0, indpb=1.0/NDIM)
    from deap.benchmarks.tools import diversity, convergence
    #from deap.benchmarks.tools.diversity
    #import deap.benchmarks.tools as tools
    from deap import tools
    pf = tools.ParetoFront()
    return toolbox, tools, history, creator, base, pf






def dt_to_ind(dtc,td):
    '''
    Re instanting data transport container at every update dtcpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''
    ind =[]
    for k in td.keys():
        ind.append(dtc.attrs[td[k]])
    ind.rheobase = dtc.rheobase
    return ind

def difference(observation,prediction): # v is a tesst
    import quantities as pq
    import numpy as np

    # The trick is.
    # prediction always has value. but observation 7 out of 8 times has mean.

    if 'value' in prediction.keys():
        unit_predictions = prediction['value']
        if 'mean' in observation.keys():
            unit_observations = observation['mean']
        elif 'value' in observation.keys():
            unit_observations = observation['value']

    if 'mean' in prediction.keys():
        unit_predictions = prediction['mean']
        if 'mean' in observation.keys():
            unit_observations = observation['mean']
        elif 'value' in observation.keys():
            unit_observations = observation['value']


    to_r_s = unit_observations.units
    unit_predictions = unit_predictions.rescale(to_r_s)
    #unit_observations = unit_observations.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )

    ##
    # Repurposed from from sciunit/sciunit/scores.py
    # line 156
    ##
    assert type(observation) in [dict,float,int,pq.Quantity]
    assert type(prediction) in [dict,float,int,pq.Quantity]
    ratio = unit_predictions / unit_observations
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
    return float(unit_delta), ratio
