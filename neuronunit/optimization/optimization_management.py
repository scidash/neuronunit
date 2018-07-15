#import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
#matplotlib.use('agg')

import numpy as np
import dask.bag as db
import pandas as pd
# Import get_neab has to happen exactly here. It has to be called only on
from neuronunit import tests
from neuronunit.optimization import get_neab
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
import numpy
from neuronunit.optimization import model_parameters as modelp
from itertools import repeat
import copy
import math

import quantities as pq
import numpy as np
from pyneuroml import pynml

from deap import base
from neuronunit.optimization.data_transport_container import DataTC
from neuronunit.models.interfaces import glif


import os
import pickle
from itertools import repeat
import neuronunit
import multiprocessing
npartitions = multiprocessing.cpu_count()
from collections import Iterable


class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None

        super(WSListIndividual, self).__init__(*args, **kwargs)



class WSFloatIndividual(float):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None

        super(WSFloatIndividual, self).__init__()

def write_opt_to_nml(path,param_dict):
    '''
    Write optimimal simulation parameters back to NeuroML.
    '''
    orig_lems_file_path = path_params['model_path']
    more_attributes = pynml.read_lems_file(orig_lems_file_path,
                                           include_includes=True,
                                           debug=False)
    for i in more_attributes.components:
        new = {}
        if str('izhikevich2007Cell') in i.type:
            for k,v in i.parameters.items():
                units = v.split()
                if len(units) == 2:
                    units = units[1]
                else:
                    units = 'mV'
                new[k] = str(param_dict[k]) + str(' ') + str(units)
            i.parameters = new
    fopen = open(path+'.nml','w')
    more_attributes.export_to_file(fopen)
    fopen.close()
    return

def dtc_to_rheo(xargs):
    dtc,rtest,backend = xargs
    LEMS_MODEL_PATH = path_params['model_path']
    model = ReducedModel(LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    dtc.scores = {}
    dtc.score = {}
    score = rtest.judge(model,stop_on_error = False, deep_error = False)
    if score.sort_key is not None:
        dtc.scores.get(str(rtest), 1 - score.sort_key)
        dtc.scores[str(rtest)] = 1 - score.sort_key
        dtc = score_proc(dtc,rtest,score)
        dtc.rheobase = score.prediction
    return dtc

def score_proc(dtc,t,score):
    dtc.score[str(t)] = {}
    dtc.score[str(t)]['value'] = copy.copy(score.sort_key)
    if hasattr(score,'prediction'):
        if type(score.prediction) is not type(None):
            dtc.score[str(t)][str('prediction')] = score.prediction
            dtc.score[str(t)][str('observation')] = score.observation
            boolean_means = bool('mean' in score.observation.keys() and 'mean' in score.prediction.keys())
            boolean_value = bool('value' in score.observation.keys() and 'value' in score.prediction.keys())
            if boolean_means:
                dtc.score[str(t)][str('agreement')] = np.abs(score.observation['mean'] - score.prediction['mean'])
            if boolean_value:
                dtc.score[str(t)][str('agreement')] = np.abs(score.observation['value'] - score.prediction['value'])

    return dtc

    #if score.sort_key is not None:
    #    dtc.scores[str(t)] = 1.0 - score.sort_key
    #else:
    #    dtc.scores[str(t)] = 1.0

def nunit_evaluation(tuple_object):#,backend=None):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    dtc,tests = tuple_object
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    LEMS_MODEL_PATH = path_params['model_path']
    assert dtc.rheobase is not None
    backend = dtc.backend
    if backend == 'glif':
        model = glif.GC()#ReducedModel(LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON',{'DTC':dtc}))
        tests[0].prediction = dtc.rheobase
        model.rheobase = dtc.rheobase['value']
    else:
        model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('NEURON',{'DTC':dtc}))
        model.set_attrs(dtc.attrs)
        tests[0].prediction = dtc.rheobase
        model.rheobase = dtc.rheobase['value']
    tests = [t for t in tests if str('RheobaseTestP') not in str(t) ]
    for k,t in enumerate(tests):
        t.params = dtc.vtest[k]
        score = None
        #model = None
        #model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('NEURON',{'DTC':dtc}))
        model.set_attrs(dtc.attrs)
        score = t.judge(model,stop_on_error = False, deep_error = False)
        #print(score)
        if score.sort_key is not None:
            dtc.scores.get(str(t), 1 - score.sort_key)
            dtc.scores[str(t)] = 1 - score.sort_key

            dtc = score_proc(dtc,t,copy.copy(score))
        else:
            dtc.scores[str(t)] = 1.0
        #import pdb; pdb.set_trace()
        print(dtc.scores)
    assert len(dtc.scores.keys()) >= 2
    return dtc

def evaluate(dtc):
    fitness = [ 1.0 for i in range(0,len(dtc.scores.keys())) ]
    print(len(fitness))
    for k,t in enumerate(dtc.scores.keys()):
        fitness[k] = dtc.scores[str(t)]
    return tuple(fitness,)

def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list

def format_test(xargs):
    '''
    pre format the current injection dictionary based on pre computed
    rheobase values of current injection.
    This is much like the hooked method from the old get neab file.
    '''
    dtc,tests = xargs
    dtc.vtest = None
    dtc.vtest = {}

    for k,v in enumerate(tests):
        dtc.vtest[k] = {}
        #dtc.vtest.get(k,{})
        dtc.vtest[k]['injected_square_current'] = {}
        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            dtc.vtest[k]['injected_square_current']['duration'] = 100 * pq.ms
            dtc.vtest[k]['injected_square_current']['amplitude'] = -10 *pq.pA
            dtc.vtest[k]['injected_square_current']['delay'] = 30 * pq.ms

        if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
            # Threshold current.
            dtc.vtest[k]['injected_square_current']['duration'] = 1000 * pq.ms
            dtc.vtest[k]['injected_square_current']['amplitude'] = dtc.rheobase['value']
            dtc.vtest[k]['injected_square_current']['delay'] = 250 * pq.ms # + 150
    return dtc

def update_dtc_pop(pop, td, backend = None):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    toolbox = base.Toolbox()
    pop = [toolbox.clone(i) for i in pop ]

    def transform(ind):
        # The merits of defining a function in a function
        # is that it yields a semi global scoped variables.
        #
        dtc = DataTC()
        LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
        if backend is not None:
            dtc.backend = backend
        else:
            dtc.backend = 'NEURON'

        dtc.attrs = {}
        if isinstance(ind, Iterable):
            for i,j in enumerate(ind):
                dtc.attrs[str(td[i])] = j
        else:
            dtc.attrs[str(td[0])] = ind
        dtc.evaluated = False
        return dtc

    if len(pop) > 1:
        npart = np.min([multiprocessing.cpu_count(),len(pop)])
        bag = db.from_sequence(pop, npartitions = npart)
        dtcpop = list(bag.map(transform).compute())

    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = list(transform(pop))
        assert len(dtcpop) == len(pop)
    return dtcpop


def run_ga(model_params,nparams,npoints,test, provided_keys = None):
    # https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python
    # These imports need to be defined with local scope to avoid circular importing problems
    # Try to fix local imports later.
    from bluepyopt.deapext.optimisations import DEAPOptimisation
    from neuronunit.optimization.exhaustive_search import create_grid
    from neuronunit.optimization.exhaustive_search import reduce_params
    if type(provided_keys) is not type(None):
        model_params = { k:model_params[k] for k in provided_keys }
    subset = reduce_params(model_params,nparams)
    MU = int(np.floor(npoints))
    max_ngen = int(np.floor(nparams))
    #assert (MU * max_ngen) < (npoints * nparams)
    DO = DEAPOptimisation(offspring_size = MU, error_criterion = test, selection = str('selNSGA'), provided_dict = subset, elite_size = 2)
    #assert len(DO.params.items()) == 3
    ga_out = DO.run(offspring_size = MU, max_ngen = 15)
    with open('all_ga_cell.p','wb') as f:
        pickle.dump(ga_out,f)
    return ga_out

def rheobase(pop, td, rt):
    if not hasattr(pop[0],'rheobase'):
        pop = [ WSFloatIndividual(ind) for ind in pop if type(ind) is not type(list) ]
    dtcpop = update_dtc_pop(pop, td)
    #if isinstance(dtcpop, Iterable):
    dtcpop = iter(dtcpop)
    xargs = iter(zip(dtcpop,repeat(rt),repeat('NEURON')))

    dtcpop = list(map(dtc_to_rheo,xargs))
    for ind,d in zip(pop,dtcpop):
        ind.rheobase = d.rheobase
    # Done changed the score away from Ratio to Z.
    return pop, dtcpop


def update_deap_pop(pop, tests, td, backend = None):
    '''
    Inputs a population of genes (pop).
    Returned neuronunit scored DTCs (dtcpop).
    This method converts a population of genes to a population of Data Transport Containers,
    Which act as communicatable data types for storing model attributes.
    Rheobase values are found on the DTCs
    DTCs for which a rheobase value of x (pA)<=0 are filtered out
    DTCs are then scored by neuronunit, using neuronunit models that act in place.
    '''
    # Rheobase value obtainment.

    dtcpop = None
    pop, dtcpop = rheobase(copy.copy(pop), td, tests[0])
    dtcpop = copy.copy(dtcpop)
    # NeuronUnit testing

    xargs = zip(dtcpop,repeat(tests))
    dtcpop = list(map(format_test,xargs))
    npart = np.min([multiprocessing.cpu_count(),len(pop)])
    dtcbag = db.from_sequence(list(zip(dtcpop,repeat(tests))), npartitions = npart)

    dtcpop = list(dtcbag.map(nunit_evaluation).compute())

    for i,d in enumerate(dtcpop):
        pop[i].dtc = None
        pop[i].dtc = copy.copy(dtcpop[i])
    invalid_dtc_not = [ i for i in pop if not hasattr(i,'dtc') ]
    if len(invalid_dtc_not) !=0:
        import pdb; pdb.set_trace()
    return pop


def create_subset(nparams = 10, provided_dict = None):
    if type(provided_dict) is type(None):
        mp = modelp.model_params
        key_list = list(mp.keys())
        reduced_key_list = key_list[0:nparams]
    else:
        key_list = list(provided_dict.keys())
        reduced_key_list = key_list[0:nparams]

    subset = { k:provided_dict[k] for k in reduced_key_list }
    return subset
