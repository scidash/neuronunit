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
from sklearn.model_selection import ParameterGrid
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



def add_constant(hold_constant,consumable_,td):
    hc = list(hold_constant.values())
    for c in consumable_:
        if type(c) is type(dict()):
            for k,v in hold_constant.items():
                c[k] = v
    for k in hold_constant.keys():
        td.append(k)
    return td, consumable_


# Intended to replace code in grid
def build_grid(means,stds=None,boundaries=None,k=5,exponential=True):
    """Generate a list of dictionaries corresponding to grid points to search."""
    param_names = list(means.keys())
    assert (stds or boundaries) and not (stds and boundaries), "Must provide stds or boundaries, but not both"
    param_names = means.keys()
    if stds or boundaries:

        if exponential:
            boundaries = {key:(means[key] - stds[key]*(2**k),means[key] + stds[key]*(2**k)) for key in param_names}
        elif boundaries == None:
            boundaries = {key:(means[key] - stds[key]*k,means[key] + stds[key]*k) for key in param_names}
        else:
            pass

        grids = {p:[] for p in param_names}
        #print(means,stds,boundaries_left,boundaries_right)
        for p in param_names:
            if exponential:
                grids[p] += [means[p] - (means[p]-boundaries[p][0])/(2**kj) for kj in range(0,k+1,1)]
                grids[p] += [means[p]]
                grids[p] += [means[p] + (boundaries[p][1]-means[p])/(2**kj) for kj in range(k,-1,-1)]
            else:
                grids[p] += [means[p] - (kj+1)*(means[p]-boundaries[p][0])/k for kj in range(0,k+1,1)]
                grids[p] += [means[p]]
                grids[p] += [means[p] + (kj+1)*(boundaries[p][1]-means[p])/k for kj in range(k,-1,-1)]
        return list(ParameterGrid(grids))

def build_grid_wrapper(means,stds=None,boundaries=None,k=5):
    """Generate a list of dictionaries corresponding to grid points
    where all but one parameter is held constant"""
    grid = build_grid(means,stds=stds,boundaries=boundaries,k=k)
    grid = [x for x in grid if sum(x[key]==means[key] for key in means)>=(len(means)-1)]
    return grid


def hack_judge(test_and_models):
    LEMS_MODEL_PATH = path_params['model_path']
    (test, dtc) = test_and_models
    model = None
    obs = test.observation
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
    model.set_attrs(dtc.attrs)
    pred = test.generate_prediction(model)
    score = test.compute_score(obs,pred)
    return score, pred

def dtc_to_rheo(dtc):
    rtest = dtc.rtest
    dtc.scores = {}
    dtc.score = {}
    LEMS_MODEL_PATH = path_params['model_path']
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
    model.set_attrs(dtc.attrs)
    dtc.rheobase = rtest.generate_prediction(model)
    obs = rtest.observation
    score = rtest.compute_score(obs,dtc.rheobase)
    dtc.scores[str(rtest)] = score.sort_key
    return dtc


#@jit
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


#@jit
def format_test(dtc):
    '''
    pre format the current injection dictionary based on pre computed
    rheobase values of current injection.
    This is much like the hooked method from the old get neab file.
    '''
    dtc.vtest = {}
    for k,v in enumerate(dtc.tests):
        dtc.vtest[k] = {}
        dtc.vtest[k]['injected_square_current'] = {}
        if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
            DURATION = 1000.0*pq.ms
            DELAY = 100.0*pq.ms
            dtc.vtest[k]['injected_square_current']['delay']= DELAY
            dtc.vtest[k]['injected_square_current']['duration'] = DURATION
            dtc.vtest[k]['injected_square_current']['amplitude'] = dtc.rheobase
        else:
            DURATION = 500.0*pq.ms
            DELAY = 200.0*pq.ms
            dtc.vtest[k]['injected_square_current'] = {}
            dtc.vtest[k]['injected_square_current']['delay']= DELAY
            dtc.vtest[k]['injected_square_current']['duration'] = DURATION
            dtc.vtest[k]['injected_square_current']['amplitude'] = -10*pq.pA

        # not returned so actually not effective
        #v.params = dtc.vest[k]

    return dtc


import dask.bag as db
# The rheobase has been obtained seperately and cannot be db mapped.
# Nested DB mappings dont work.
from itertools import repeat


def nunit_evaluation(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    #dtc,tests = tuple_object
    tests = dtc.tests
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    LEMS_MODEL_PATH = path_params['model_path']
    assert dtc.rheobase is not None
    if dtc.rheobase == 1:
        return

    for k,t in enumerate(tests[1::]):
        t.params = dtc.vtest[k]
        score,_ = hack_judge((t,dtc))
        #score = t.judge(model,stop_on_error = False, deep_error = True)
        if type(score.sort_key) is not type(None):
            dtc.scores[str(t)] = 1 - score.sort_key
        else:
            dtc.scores[str(t)] = 1.0
        dtc = score_proc(dtc,t,copy.copy(score))
    dtc.get_ss() # compute the sum of sciunit score components.
    return dtc




def evaluate(dtc):
    error_length = len(dtc.scores.keys())
    fitness = [ 1.0 for i in range(0,error_length) ]
    other_fitness = [ 1.0 for i in range(0,error_length) ]
    for k,t in enumerate(dtc.scores.keys()):
        fitness[k] = dtc.scores[str(t)]
        #if str('agreement') in dtc.score[str(t)].keys():
        #    other_fitness[k] = dtc.score[str(t)]['agreement']
    return tuple(fitness,)

def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list


from itertools import repeat

def transform(xargs):
    (ind,td,backend) = xargs

    # The merits of defining a function in a function
    # is that it yields a semi global scoped variables.
    # conisider refactoring outer function as a decorator.
    dtc = DataTC()

    LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
    dtc.backend = 'RAW'
    dtc.attrs = {}
    for i,j in enumerate(ind):
        dtc.attrs[str(td[i])] = j

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

    if isinstance(pop, Iterable):# and type(pop[0]) is not type(str('')):
        xargs = zip(pop,repeat(td),repeat(backend))
        npart = np.min([multiprocessing.cpu_count(),len(pop)])
        bag = db.from_sequence(xargs, npartitions = npart)
        dtcpop = list(bag.map(transform).compute())
        assert len(dtcpop) == len(pop)
    else:
        # TODO
        # erradicate xargs.
        # everything can be stored in the DTC,
        # so stop making complicated tuple xargs patterns.
        for p in pop:
            p.td = td
            p.backend = str('RAW')
        # above replaces need for this line:
        xargs = (pop,td,repeat(backend))

        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = [ transform(xargs) ]
        assert dtcpop[0].backend is 'RAW'
    print(dtcpop)
    return dtcpop

#@jit
def run_ga(model_params, max_ngen, test, free_params = None, hc = None):
    # https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python
    # These imports need to be defined with local scope to avoid circular importing problems
    # Try to fix local imports later.
    from bluepyopt.deapext.optimisations import SciUnitOptimization
    ss = {}
    for k in free_params:
        ss[k] = model_params[k]

    MU = 2**len(list(free_params))
    MU = int(np.floor(MU/2))
    #MU = 20
    max_ngen = int(np.floor(max_ngen))
    selection = str('selNSGA')
    DO = SciUnitOptimization(offspring_size = MU, error_criterion = test, selection = selection, boundary_dict = ss, elite_size = 2)

    ga_out = DO.run(offspring_size = MU, max_ngen = max_ngen)
    ga_out['dhof'] = [ h.dtc for h in ga_out['hof'] ]
    ga_out['dbest'] = ga_out['hof'][0].dtc

    return ga_out, DO

def rheobase_old(pop, td, rt):
    '''
    Calculate rheobase for a given population pop
    Ordered parameter dictionary td
    and rheobase test rt
    '''
    from neuronunit.optimization.exhaustive_search import update_dtc_grid
    dtcpop = list(update_dtc_pop(pop, td))
    for d in dtcpop:
        d.rtest = rt
        d.backend = str('RAW')

    dtcpop = list(map(dtc_to_rheo,dtcpop))
    for ind,d in zip(pop,dtcpop):
        ind.rheobase = d.rheobase['value']
        d.rheobase = d.rheobase['value']
    return pop, dtcpop



#@jit
def impute_check(pop,dtcpop,td,tests):
    delta = len(pop) - len(dtcpop)
    # at this point we want to take means of all the genes that are not deleted.

    # if a rheobase value cannot be found for a given set of dtc model more_attributes
    # delete that model, or rather, filter it out above, and impute
    # a new model from the mean of the pre-existing model attributes.
    impute_pop = []
    if delta != 0:
        impute = []
        impute_gene = [] # impute individual, not impute index
        for t in td:
             impute_gene.append(np.mean([ d.attrs[t] for d in dtcpop ]))

        ind = WSListIndividual(impute_gene)
        # newest functioning code.
        # other broken transform should be modelled on this.
        ## what function transform should consist of.
        dtc = DataTC()
        LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
        dtc.backend = 'RAW'
        dtc.attrs = {}

        for i,j in enumerate(ind):
            dtc.attrs[str(td[i])] = j

        ## end function transform

        dtc.rtest = tests[0]
        dtc.backend = str('RAW')
        dtc = dtc_to_rheo(dtc)
        ind.rheobase = dtc.rheobase
        print(dtc.attrs,dtc.rheobase,'still failing')
        if type(ind.rheobase) != 1.0:
            #pop.append(ind)
            dtcpop.append(dtc)
    return dtcpop,pop





def serial_route(pop,td,tests):
    #pop, dtc = rheobase_old(copy.copy(pop), td, tests[0])
    if type(dtc.rheobase) is type(None):
        print('Error Score bad model')
        for t in tests:
            dtc.scores = {}
            dtc.get_ss()

    else:
        dtc = format_test((dtc,tests))
        dtc = nunit_evaluation((dtc,tests))
        print(dtc.get_ss())
    return pop, dtc

def parallel_route(pop,dtcpop,tests,td):
    print([type(d) for d in dtcpop])
    print([d for d in dtcpop])
    for d in dtcpop:
        d.tests = tests

    dtcpop = list(map(format_test,dtcpop))
    #import pdb; pdb.set_trace()
    npart = np.min([multiprocessing.cpu_count(),len(pop)])
    dtcbag = db.from_sequence(dtcpop, npartitions = npart)
    dtcpop = list(dtcbag.map(nunit_evaluation).compute())
    #for zipped in zip(dtcpop,repeat(tests)):
    #    junk = nunit_evaluation(zipped)
    #import pdb; pdb.set_trace()
    for i,d in enumerate(dtcpop):
        if not hasattr(pop[i],'dtc'):
            pop[i] = WSListIndividual(pop[i])
            pop[i].dtc = None

        pop[i].dtc = copy.copy(dtcpop[i])
        pop[i].dtc.get_ss()
    invalid_dtc_not = [ i for i in pop if not hasattr(i,'dtc') ]

    return pop, dtcpop

def test_runner(pop,td,tests):
    pop, dtcpop = rheobase_old(pop, td, tests[0])
    # parallel route:
    #dtcpop = [ dtc for dtc in dtcpop if hasattr(dtc,'rheobase') ]
    dtcpop = [ dtc for dtc in dtcpop if not isinstance(dtc.rheobase,type(None)) ]
    dtcpop = [ dtc for dtc in dtcpop if dtc.rheobase!=1.0 ]
    before = len(pop)
    pop = [ p for p in pop if p.rheobase!=1.0 ]
    after = len(pop)

    print([dtc.rheobase for dtc in dtcpop])

    while before>after:
        dtcpop,pop = impute_check(pop,dtcpop,td,tests)
        before = len(pop)

    #import pdb; pdb.set_trace()
    pop,dtcpop = parallel_route(pop,dtcpop,tests,td)
    for p,d in zip(pop,dtcpop):
        p.dtc = d
    print([p.dtc for p in pop])

    if isinstance(pop, Iterable) and isinstance(dtcpop,Iterable):

        for p,d in zip(pop,dtcpop):
            if type(p) is type(None) or type(d) is type(None):
                import pdb
                pdb.set_trace()


    #serial route:
    if not isinstance(pop, Iterable):# and not isinstance(dtcpop,Iterable):
        pop,dtcpop = serial_route(pop,td,tests)
        print('serial badness')

    print('tests, completed, now gene computations')
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
    print('not even to test runner getting')
    pop, dtcpop = test_runner(pop,td,tests)
    for p,d in zip(pop,dtcpop):
        p.dtc = d
    if not isinstance(pop, Iterable) and not isinstance(dtcpop,Iterable):
        pop.dtc = dtcpop
        if type(pop.dtc.rheobase) is type(None):
            pop.dtc.scores = {}
            for t in tests:
                pop.dtc.scores[str(t)] = 1.0
        print(pop.dtc.get_ss())
    else:
        pass
    return pop

def create_subset(nparams = 10, boundary_dict = None):
    # used by GA to find subsets in parameter space.
    if type(boundary_dict) is type(None):
        raise ValueError('A dictionary was not not supplied and a specific bad thing happened.')

        mp = modelp.model_params
        key_list = list(mp.keys())
        reduced_key_list = key_list[0:nparams]
    else:
        key_list = list(boundary_dict.keys())
        reduced_key_list = key_list[0:nparams]
    subset = { k:boundary_dict[k] for k in reduced_key_list }
    return subset
