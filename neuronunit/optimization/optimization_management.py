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
from numba import jit

# DEAP mutation strategies:
# https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.mutESLogNormal
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

@jit
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
    print(dtc.attrs)
    import pdb
    pdb.set_trace()
    model.set_attrs(dtc.attrs)
    dtc.scores = {}
    dtc.score = {}
    score = rtest.judge(model,stop_on_error = False, deep_error = False)
    print(score)
    dtc.scores.get(str(rtest), 1.0)            
    if hasattr(score,'prediction'):
        has_pred = bool(type(score.prediction) is not type(None))
        has_zf = bool(type(score.sort_key) is not type(None))
        if has_zf and has_pred:
            dtc.scores[str(rtest)] = 1 - score.sort_key
            dtc = score_proc(dtc,rtest,score)
            dtc.rheobase = score.prediction
    else:
        dtc.rheobase = None
        
    return dtc
    
@jit
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


def nunit_evaluation(tuple_object):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    dtc,tests = tuple_object
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    LEMS_MODEL_PATH = path_params['model_path']
    assert dtc.rheobase is not None
    tests = [ t for t in tests if str('RheobaseTestP') not in str(t) ]

    for k,t in enumerate(tests):
        # find sciunit scores for tests passed in.
        t.params = dtc.vtest[k]
        model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('NEURON',{'DTC':dtc}))
        model.set_attrs(dtc.attrs)
        score = t.judge(model,stop_on_error = False, deep_error = False)        
        if type(score.sort_key) is not type(None):
            dtc.scores[str(t)] = 1 - score.sort_key
        else:
            dtc.scores[str(t)] = 1.0
        dtc = score_proc(dtc,t,copy.copy(score))
    dtc.get_ss() # compute the sum of sciunit score components.
    return dtc

    '''
    Consider factor glif model option back in via function decorator.
    backend = dtc.backend

    if backend == 'glif':
        model = glif.GC()
        tests[0].prediction = dtc.rheobase
        model.rheobase = dtc.rheobase['value']
    else:
        model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('NEURON',{'DTC':dtc}))
        model.set_attrs(dtc.attrs)
        tests[0].prediction = dtc.rheobase
        model.rheobase = dtc.rheobase['value']
    '''
    #return model


def evaluate(dtc):
    error_length = len(dtc.scores.keys())
    fitness = [ 1.0 for i in range(0,error_length) ]
    old_fitness = [ 1.0 for i in range(0,error_length) ]
    
    for k,t in enumerate(dtc.scores.keys()):
        old_fitness[k] = dtc.scores[str(t)]
        if str('agreement') in dtc.score[str(t)].keys():
            fitness[k] = dtc.score[str(t)]['agreement']

    return tuple(fitness,)

def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list

@jit
def format_test(xargs):
    '''
    pre format the current injection dictionary based on pre computed
    rheobase values of current injection.
    This is much like the hooked method from the old get neab file.
    '''
    dtc,tests = xargs
    dtc.vtest = {}
    for k,v in enumerate(tests):
        dtc.vtest[k] = {}
        dtc.vtest[k]['injected_square_current'] = {}
        if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
            dtc.vtest[k]['injected_square_current']['amplitude'] = dtc.rheobase['value']

            # Threshold current.
            #dtc.vtest[k]['injected_square_current']['duration'] = 1000 * pq.ms
            #dtc.vtest[k]['injected_square_current']['delay'] = 250 * pq.ms # + 150
    return dtc

'''
if k == 1 or k == 2 or k == 3:
    # Negative square pulse current.
    dtc.vtest[k]['injected_square_current']['duration'] = 100 * pq.ms
    dtc.vtest[k]['injected_square_current']['amplitude'] = -10 *pq.pA
    dtc.vtest[k]['injected_square_current']['delay'] = 30 * pq.ms
'''

from itertools import repeat

def transform(xargs):
    ind,td,backend = xargs
        
    # The merits of defining a function in a function
    # is that it yields a semi global scoped variables.
    # conisider refactoring outer function as a decorator.
    dtc = DataTC()
    LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
    if backend is not None:
        dtc.backend = backend
    else:
        dtc.backend = 'NEURON'
    dtc.attrs = {}
        
    if isinstance(td, list) and isinstance(ind,Iterable):
        for i,j in enumerate(ind):
            dtc.attrs[str(td[i])] = j
    else:
        td = td[0]
        dtc.attrs[td] = ind
        
        # remember a string is iterable
    dtc.evaluated = False
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

    if isinstance(pop, list) and type(pop[0]) is not type(str('')):
        pop = [toolbox.clone(i) for i in pop ]
        xargs = zip(pop,repeat(td),repeat(backend))

        npart = np.min([multiprocessing.cpu_count(),len(pop)])
        bag = db.from_sequence(xargs, npartitions = npart)
        dtcpop = list(bag.map(transform).compute())
        assert len(dtcpop) == len(pop)
    else:
        pop = [toolbox.clone(i) for i in pop ]
        xargs = (pop,td,repeat(backend))
        print(xargs)
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = [ transform(xargs) ]
    
        assert dtcpop[0].backend is 'NEURON'
    return dtcpop

#@jit
def run_ga(model_params,npoints,test, provided_keys = None, nr = None):
    # https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python
    # These imports need to be defined with local scope to avoid circular importing problems
    # Try to fix local imports later.
    # from bluepyopt.deapext.optimisations import SciUnitOptimization
    from neuronunit.optimization.bp_opt import SciUnitOptimization
    from neuronunit.optimization.exhaustive_search import create_grid
    from neuronunit.optimization.exhaustive_search import reduce_params

    ss = {}
    for k in provided_keys:
        ss[k] = model_params[k]
    MU = int(np.floor(npoints))
    max_ngen = int(np.floor(npoints))
    selection = str('selNSGA')
    DO = SciUnitOptimization(offspring_size = MU, error_criterion = test, selection = selection, provided_dict = ss, elite_size = 2)
    ga_out = DO.run(offspring_size = MU, max_ngen = max_ngen)
    return ga_out, DO


def blank_slate(dtcpop,rt):
    # not used
    for d in dtcpop:
        for t in tests:    
            d.scores.get(str(t), 1.0)
    return dtcpop

def sanity_check_score(pop,td,tests):
    '''
    Used for debugging with mock-up models
    Not used
    '''
    
    dtcpop = update_dtc_pop(pop,td)
    for dtc in dtcpop:
        dtc.scores = None
        dtc.scores = {}

    for t in tests:
        for dtc in dtcpop:
            LEMS_MODEL_PATH = path_params['model_path']
            model = ReducedModel(LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
            model.set_attrs(dtc.attrs)
            score = t.judge(model)
            dtc.scores.get(str(t), 1.0)
            
            if score.sort_key is not None:
                dtc.scores[str(t)] = 1 - score.sort_key
    return dtcpop

def rheobase(pop, td, rt):
    '''
    Calculate rheobase for a given population pop
    Ordered parameter dictionary td 
    and rheobase test rt
    '''
    from neuronunit.optimization.exhaustive_search import update_dtc_grid 
    
    if not isinstance(pop, Iterable):
        # WSFloat a DEAP BluePyOpt extensible Float type.
        pop = WSFloatIndividual(pop) 
        dtc = update_dtc_grid(pop, td)
        if isinstance(dtc, Iterable):
            xargs = (dtc[0],rt,str('NEURON')) 
            dtc = dtc_to_rheo(xargs)
            pop.rheobase = dtc.rheobase
            return pop,dtc

        else:
            xargs = (dtc,rt,str('NEURON')) 
            dtc = dtc_to_rheo(xargs)
            pop.rheobase = dtc.rheobase
            return pop,dtc
    else:
        dtcpop = list(update_dtc_pop(pop, td))
        xargs = iter(zip(dtcpop,repeat(rt),repeat('NEURON')))
        dtcpop = list(map(dtc_to_rheo,xargs))
        pop = [ WSFloatIndividual(ind) for ind in pop ]
        for ind,d in zip(pop,dtcpop):
            ind.rheobase = d.rheobase
        return pop, dtcpop
     



#@jit
def impute_check(pop,dtcpop):
    delta0 = len(pop) - len(dtcpop)
    delta1 = len([d for d in dtcpop if type(d.rheobase) is type(None) ])
    # if a rheobase value cannot be found for a given set of dtc model more_attributes
    # delete that model, or rather, filter it out above, and impute
    # a new model from the mean of the pre-existing model attributes.
    impute_pop = []
    #if delta != 0:
    if delta0 or delta1:
        impute = []
        # iterate across attributes in the gene.
        for i in range(0,len(pop[0])):
            # create a gene that has attributes that are mean values.
            impute_ind.append(np.mean([ p[i] for p in pop ]))
        # cast type from list to formal DEAP/BPO type
        if len(impute_ind) == 1:
            impute_ind = WSFloatIndividual(impute_ind)
        else:
            impute_ind = WSListIndividual(impute_ind)
        for d in range(0,delta):
            temp = copy.copy(dtcpop[0])
            for i,k in enumerate(temp.attrs.keys()):
                temp.attrs[k] = impute_ind[i]

            impute_ind, temp = rheobase(impute_ind, td, tests[0])
            
            pop.append(impute_ind)
            dtcpop.append(temp)
    return dtcpop,pop

def serial_route(pop,td,tests):
    pop, dtc = rheobase(copy.copy(pop), td, tests[0])
    if type(dtc.rheobase) is type(None):
        print('Error Score bad model')
        for t in tests:
            dtc.scores = {}
            dtc.scores.get(t,1.0)
            dtc.get_ss()
            
    else:
        dtc = format_test((dtc,tests))
        dtc = nunit_evaluation((dtc,tests))
        dtc.get_ss()
    return pop, dtc

def parallel_route(pop,dtcpop,tests):
    print('main parallel entry point \n\n\n\n\n')
    #dtcpop,pop = impute_check(copy.copy(pop),dtcpop)
    #import pdb
    #print(dtcpop,pop)
    #pdb.set_trace()
    xargs = zip(dtcpop,repeat(tests))
    print('len tests {0} tests {1}'.format(len(tests),tests))

    dtcpop = list(map(format_test,xargs))
    npart = np.min([multiprocessing.cpu_count(),len(pop)])
    dtcbag = db.from_sequence(list(zip(dtcpop,repeat(tests))), npartitions = npart)
    dtcpop = list(dtcbag.map(nunit_evaluation).compute())
    for i,d in enumerate(dtcpop):
        pop[i].dtc = copy.copy(dtcpop[i])
        pop[i].dtc.get_ss()
    invalid_dtc_not = [ i for i in pop if not hasattr(i,'dtc') ]
    return pop, dtcpop

def test_runner(pop,td,tests):
    
    # NeuronUnit testing
    pop, dtcpop = rheobase(pop, td, tests[0])
    # parallel route:

    if isinstance(pop, Iterable) and isinstance(dtcpop,Iterable):

        for p,d in zip(pop,dtcpop):
            if type(p) is type(None) or type(d) is type(None):
                import pdb
                pdb.set_trace()

        pop,dtcpop = parallel_route(pop,dtcpop,tests)
    #serial route:
    if not isinstance(pop, Iterable):# and not isinstance(dtcpop,Iterable):
        pop,dtcpop = serial_route(pop,td,tests)
    return pop,dtcpop    

    
def update_deap_pop(pop, tests, td, backend = None):
    '''
    # Inputs a population of genes (pop).
    Returned neuronunit scored DataTransportContainers (dtcpop).
    This method converts a population of genes to a population of Data Transport Containers,
    Which act as communicatable data types for storing model attributes.
    Rheobase values are found on the DTCs
    DTCs for which a rheobase value of x (pA)<=0 are filtered out
    DTCs are then scored by neuronunit, using neuronunit models that act in place.
    '''
    pop = copy.copy(pop)
    pop, dtcpop = test_runner(pop,td,tests)
    if not isinstance(pop, Iterable) or not isinstance(dtcpop,Iterable):
        pop.dtc = dtcpop
        if type(pop.dtc.rheobase) is type(None):
            pop.dtc.scores = {}
            for t in tests:
                pop.dtc.scores[str(t)] = 1.0
        print(pop.dtc.get_ss())
    return pop

def create_subset(nparams = 10, provided_dict = None):
    # used by GA to find subsets in parameter space.
    if type(provided_dict) is type(None):
        raise ValueError('A dictionary was not not supplied and a specific bad thing happened.')
    
        mp = modelp.model_params
        key_list = list(mp.keys())
        reduced_key_list = key_list[0:nparams]
    else:
        key_list = list(provided_dict.keys())
        reduced_key_list = key_list[0:nparams]
    subset = { k:provided_dict[k] for k in reduced_key_list }
    return subset
