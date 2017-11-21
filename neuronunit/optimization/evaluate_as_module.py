##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##



import os
from neuronunit.models import backends
import neuronunit
print(neuronunit.models.__file__)
from neuronunit.models.reduced import ReducedModel
from ipyparallel import depend, require, dependent
import ipyparallel as ipp

rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]
'''
from neuronunit.optimization import get_neab
import ipyparallel as ipp
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
'''
class Individual(object):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error=None
        self.results=None
        self.name=''
        self.attrs = {}
        self.params=None
        self.score=None
        self.fitness=None
        self.lookup={}
        self.rheobase=None
        self.fitness = creator.FitnessMin

@require('numpy, deap','random')
def import_list(ipp,subset,NDIM):
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
    weights = tuple([-1.0 for i in range(0,NDIM)])
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, len(BOUND_UP))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    return toolbox, tools, history, creator, base

def update_dtc_pop(pop, td):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    import copy
    import numpy as np
    from deap import base
    toolbox = base.Toolbox()
    Individual = ipp.Reference('Individual')
    pop = [toolbox.clone(i) for i in pop ]
    def transform(ind):
        from neuronunit.optimization.data_transport_container import DataTC
        dtc = DataTC()
        print(dtc)
        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[td[i]] = str(j)
        dtc.attrs = param_dict
        dtc.evaluated = False
        return dtc
    if len(pop) > 0:
        dtcpop = list(dview.map_sync(transform, pop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = list(transform(pop))
    return dtcpop

def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict

def dt_to_ind(dtc,td):
    '''
    Re instanting data transport container at every update dtcpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''
    ind =[]
    for k in td.keys():
        ind.append(dtc.attrs[td[k]])
    ind.append(dtc.rheobase)
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

def pre_format(dtc):
    '''
    pre format the current injection dictionary based on pre computed
    rheobase values of current injection.
    This is much like the hooked method from the old get neab file.
    '''
    import quantities as pq
    import copy
    dtc.vtest = None
    dtc.vtest = {}
    from neuronunit.optimization import get_neab
    tests = get_neab.tests
    for k,v in enumerate(tests):
        dtc.vtest[k] = {}
        dtc.vtest[k]['injected_square_current'] = {}
    for k,v in enumerate(tests):
        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            dtc.vtest[k]['injected_square_current']['duration'] = 100 * pq.ms
            dtc.vtest[k]['injected_square_current']['amplitude'] = -10 *pq.pA
            dtc.vtest[k]['injected_square_current']['delay'] = 30 * pq.ms

        if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
            # Threshold current.
            dtc.vtest[k]['injected_square_current']['duration'] = 1000 * pq.ms
            dtc.vtest[k]['injected_square_current']['amplitude'] = dtc.rheobase['value']
            dtc.vtest[k]['injected_square_current']['delay'] = 100 * pq.ms
    return dtc
