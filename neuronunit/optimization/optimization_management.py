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
#from neuronunit.models.interfaces import glif


import os
import pickle
from itertools import repeat
import neuronunit
import multiprocessing
npartitions = multiprocessing.cpu_count()
from collections import Iterable
from numba import jit
from sklearn.model_selection import ParameterGrid
from itertools import repeat
from collections import OrderedDict


import logging
logger = logging.getLogger('__main__')



from neuronunit.tests.fi import RheobaseTestP# as discovery
from neuronunit.tests.fi import RheobaseTest# as discovery

import dask.bag as db
# The rheobase has been obtained seperately and cannot be db mapped.
# Nested DB mappings dont work.
from itertools import repeat
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


def mint_generic_model(backend):
    LEMS_MODEL_PATH = path_params['model_path']
    return ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = str(backend))

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


def bridge_judge(test_and_models):
    # Temporarily patch sciunit judge code, which seems to be broken.
    #
    #
    (test, dtc) = test_and_models
    obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    pred = test.generate_prediction(model)
    if pred is not None:
        score = test.compute_score(obs,pred)
        #print(score.norm_score)
    else:
        score = None
    return score, pred

def get_rh(dtc,rtest):
    place_holder = {}
    place_holder['n'] = 86
    place_holder['mean'] = 10*pq.pA
    place_holder['std'] = 10*pq.pA
    place_holder['value'] = 10*pq.pA
    rtest = RheobaseTestP(observation=place_holder,name='a Rheobase test')
    dtc.rheobase = None
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    #model = mint_generic_model()
    dtc.rheobase = rtest.generate_prediction(model)#['value']
    if dtc.rheobase is None:
        dtc.rheobase = - 1.0
    return dtc


def dtc_to_rheo(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    dtc.scores = {}
    dtc.score = {}
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    rtest = [ t for t in dtc.tests if str('RheobaseTestP') == t.name ]


    if len(rtest):
        rtest = rtest[0]

        dtc.rheobase = rtest.generate_prediction(model)
        #print(dtc.rheobase)
        if dtc.rheobase is not None and dtc.rheobase !=-1.0:
            dtc.rheobase = dtc.rheobase['value']
            obs = rtest.observation
            score = rtest.compute_score(obs,dtc.rheobase)
            dtc.scores[str('RheobaseTestP')] = 1.0 - score.norm_score

            if dtc.score is not None:
                dtc = score_proc(dtc,rtest,copy.copy(score))

            rtest.params['injected_square_current']['amplitude'] = dtc.rheobase

        else:
            dtc.rheobase = - 1.0
            dtc.scores[str('RheobaseTestP')] = 1.0



    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
    return dtc


def score_proc(dtc,t,score):
    dtc.score[str(t)] = {}
    #print(score.keys())
    if hasattr(score,'norm_score'):
        dtc.score[str(t)]['value'] = copy.copy(score.norm_score)
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
        dtc.agreement = dtc.score
    return dtc

def switch_logic(tests):
    # move this logic into sciunit tests
    for t in tests:
        t.passive = None
        t.active = None
        active = False
        passive = False

        if str('RheobaseTest') == t.name:
            active = True
            passive = False
        elif str('RheobaseTestP') == t.name:
            active = True
            passive = False
        elif str('InjectedCurrentAPWidthTest') == t.name:
            active = True
            passive = False
        elif str('InjectedCurrentAPAmplitudeTest') == t.name:
            active = True
            passive = False
        elif str('InjectedCurrentAPThresholdTest') == t.name:
            active = True
            passive = False
        elif str('RestingPotentialTest') == t.name:
            passive = True
            active = False
        elif str('InputResistanceTest') == t.name:
            passive = True
            active = False
        elif str('TimeConstantTest') == t.name:
            passive = True
            active = False
        elif str('CapacitanceTest') == t.name:
            passive = True
            active = False
        t.passive = passive
        t.active = active
    return tests

def active_values(keyed,rheobase):
    DURATION = 1000.0*pq.ms
    DELAY = 100.0*pq.ms
    keyed['injected_square_current'] = {}
    keyed['injected_square_current']['delay']= DELAY
    keyed['injected_square_current']['duration'] = DURATION
    if type(rheobase) is type({str('k'):str('v')}):
        keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
    else:
        keyed['injected_square_current']['amplitude'] = rheobase

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
    #pre format the current injection dictionary based on pre computed
    #rheobase values of current injection.
    #This is much like the hooked method from the old get neab file.
    dtc.vtest = {}
    dtc.tests = switch_logic(dtc.tests)

    for k,v in enumerate(dtc.tests):
        dtc.vtest[k] = {}
        if v.passive == False and v.active == True:
            keyed = dtc.vtest[k]
            dtc.vtest[k] = active_values(keyed,dtc.rheobase)

        elif v.passive == True and v.active == False:
            keyed = dtc.vtest[k]
            dtc.vtest[k] = passive_values(keyed)
    return dtc



def allocate_worst(dtc,tests):
    # If the model fails tests, and cannot produce model driven data
    # Allocate the worst score available.
    for t in tests:
        dtc.scores[str(t)] = 1.0
        dtc.score[str(t)] = 1.0
    return dtc

def nunit_evaluation(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    tests = dtc.tests
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    LEMS_MODEL_PATH = path_params['model_path']


    if dtc.rheobase == -1.0 or type(dtc.rheobase) is type(None):
        dtc = allocate_worst(tests,dtc)
    else:
        for k,t in enumerate(tests):
            if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
                t.params = dtc.vtest[k]
                score,_= bridge_judge((t,dtc))
                if score is not None:
                    if score.norm_score is not None:
                        dtc.scores[str(t)] = 1.0 - score.norm_score
                        dtc = score_proc(dtc,t,copy.copy(score))
                else:
                    print('gets to None score type')
    # compute the sum of sciunit score components.
    dtc.summed = dtc.get_ss()
    return dtc




def evaluate(dtc):
    error_length = len(dtc.scores.keys())
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    fitness = [ 1.0 for i in range(0,error_length) ]
    for k,t in enumerate(dtc.scores.keys()):
        fitness[k] = dtc.scores[str(t)]
    return tuple(fitness,)

def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list



def transform(xargs):
    (ind,td,backend) = xargs
    dtc = DataTC()
    LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
    dtc.attrs = {}
    for i,j in enumerate(ind):
        dtc.attrs[str(td[i])] = j
    dtc.evaluated = False
    return dtc


def add_constant(hold_constant, pop, td):
    hold_constant = OrderedDict(hold_constant)
    for k in hold_constant.keys():
        td.append(k)
    for p in pop:
        for v in hold_constant.values():
            p.append(v)
    return pop,td

def update_dtc_pop(pop, td):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    if pop[0].backend is not None:
        _backend = pop[0].backend
    if isinstance(pop, Iterable):# and type(pop[0]) is not type(str('')):
        xargs = zip(pop,repeat(td),repeat(_backend))
        npart = np.min([multiprocessing.cpu_count(),len(pop)])
        bag = db.from_sequence(xargs, npartitions = npart)
        dtcpop = list(bag.map(transform).compute())
        assert len(dtcpop) == len(pop)
    else:
        for p in pop:
            p.td = td
            p.backend = str(_backend)
        # above replaces need for this line:
        xargs = (pop,td,repeat(backend))
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = [ transform(xargs) ]
        assert exec('dtcpop[0].backend is '+str(_backend)+')')
    return dtcpop




def run_ga(explore_edges, max_ngen, test, free_params = None, hc = None, NSGA = None, MU = None, seed_pop = None, model_type = str('RAW')):
    # seed_pop can be used to
    # to use existing models, that are good guesses at optima, as starting points for optimization.
    # https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python
    # These imports need to be defined with local scope to avoid circular importing problems
    # Try to fix local imports later.
    from bluepyopt.deapext.optimisations import SciUnitOptimization

    ss = {}
    for k in free_params:
        ss[k] = explore_edges[k]
    if type(MU) == type(None):
        MU = 2**len(list(free_params))
    # make sure that the gene population size is divisible by 4.
    if NSGA == True:
        selection = str('selNSGA')
    else:
        selection = str('selIBEA')
    max_ngen = int(np.floor(max_ngen))
    DO = SciUnitOptimization(offspring_size = MU, error_criterion = test, boundary_dict = ss, backend = model_type, hc = hc)#, selection = selection, boundary_dict = ss, elite_size = 2, hc=hc)

    if seed_pop is not None:
        # This is a re-run condition.
        DO.setnparams(nparams = len(free_params), boundary_dict = ss)

        DO.seed_pop = seed_pop
        DO.setup_deap()

    # This run condition should not need same arguments as above.
    ga_out = DO.run(max_ngen = max_ngen)#offspring_size = MU, )
    return ga_out, DO


def init_pop(pop, td, tests):

    from neuronunit.optimization.exhaustive_search import update_dtc_grid
    dtcpop = list(update_dtc_pop(pop, td))
    for d in dtcpop:
        d.tests = tests
        if hasattr(pop[0],'backend'):
            d.backend = pop[0].backend

    if hasattr(pop[0],'hc'):
        constant = pop[0].hc
        for d in dtcpop:
            if constant is not None:
                if len(constant):
                    d.constants = constant
                    d.add_constant()

    return pop, dtcpop

def obtain_rheobase(pop, td, tests):
    '''
    Calculate rheobase for a given population pop
    Ordered parameter dictionary td
    and rheobase test rt
    '''
    pop, dtcpop = init_pop(pop, td, tests)
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    for ind,d in zip(pop,dtcpop):
        if type(d.rheobase) is not type(1.0):
            ind.rheobase = d.rheobase
            d.rheobase = d.rheobase
        else:
            ind.rheobase = -1.0
            d.rheobase = -1.0
    return pop, dtcpop

def new_genes(pop,dtcpop,td):
    '''
    some times genes explored will not return
    un-usable simulation parameters
    genes who have no rheobase score
    will be discarded.

    BluePyOpt needs a stable
    gene number however

    This method finds how many genes have
    been discarded, and tries to build new genes
    from the existing distribution of gene values, by mimicing a normal random distribution
    of genes that are not deleted.
    if a rheobase value cannot be found for a given set of dtc model more_attributes
    delete that model, or rather, filter it out above, and make new genes based on
    the statistics of remaining genes.
    it's possible that they wont be good models either, so keep trying in that event.
    a new model from the mean of the pre-existing model attributes.
    '''
    impute_gene = [] # impute individual, not impute index
    ind = WSListIndividual()
    for t in td:
        mean = np.mean([ d.attrs[t] for d in dtcpop ])
        std = np.std([ d.attrs[t] for d in dtcpop ])
        sample = numpy.random.normal(loc=mean, scale=2*std, size=1)[0]
        ind.append(sample)
    dtc = DataTC()
    LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
    dtc.attrs = {}
    for i,j in enumerate(ind):
        dtc.attrs[str(td[i])] = j
    dtc.backend = dtcpop[0].backend
    dtc.tests = dtcpop[0].tests
    dtc = dtc_to_rheo(dtc)
    ind.rheobase = dtc.rheobase
    return ind,dtc


def serial_route(pop,td,tests):
    '''
    parallel list mapping only works with an iterable collection.
    Serial route is intended for single items.
    '''
    if type(dtc.rheobase) is type(None):
        print('Error Score bad model')
        for t in tests:
            dtc.scores = {}
            dtc.get_ss()
    else:
        dtc = format_test((dtc,tests))
        dtc = nunit_evaluation((dtc,tests))
    return pop, dtc

def filtered(pop,dtcpop):
    dtcpop = [ dtc for dtc in dtcpop if dtc.rheobase!=-1.0 ]
    pop = [ p for p in pop if p.rheobase!=-1.0 ]
    dtcpop = [ dtc for dtc in dtcpop if dtc.rheobase is not None ]
    pop = [ p for p in pop if p.rheobase is not None ]
    assert len(pop) == len(dtcpop)
    return (pop,dtcpop)


def parallel_route(pop,dtcpop,tests,td):
    for d in dtcpop:
        d.tests = copy.copy(tests)
    dtcpop = list(map(format_test,dtcpop))
    #import pdb; pdb.set_trace()
    npart = np.min([multiprocessing.cpu_count(),len(dtcpop)])
    dtcbag = db.from_sequence(dtcpop, npartitions = npart)
    dtcpop = list(dtcbag.map(nunit_evaluation).compute())
    for i,d in enumerate(dtcpop):
        if not hasattr(pop[i],'dtc'):
            pop[i] = WSListIndividual(pop[i])
            pop[i].dtc = None
        d.get_ss()
        pop[i].dtc = copy.copy(d)
    invalid_dtc_not = [ i for i in pop if not hasattr(i,'dtc') ]
    return pop, dtcpop

def make_up_lost(pop,dtcpop,td):
    before = len(pop)
    (pop,dtcpop) = filtered(pop,dtcpop)
    after = len(pop)
    assert after>0
    delta = before-after
    if delta:
        cnt = 0
        while cnt < delta:
            ind,dtc = new_genes(pop,dtcpop,td)
            if dtc.rheobase != -1.0:
                pop.append(ind)
                dtcpop.append(dtc)
                cnt += 1
    return pop, dtcpop

import dask.bag as db

def test_runner(pop,td,tests,single_spike=True):
    if single_spike:
        pop, dtcpop = obtain_rheobase(pop, td, tests)
        pop, dtcpop = make_up_lost(pop,dtcpop,td)
        # there are many models, which have no actual rheobase current injection value.
        # filter, filters out such models,
        # gew genes, add genes to make up for missing values.
        # delta is the number of genes to replace.

    else:
        pop, dtcpop = init_pop(pop, td, tests)
    '''
    xargs = zip(pop,dtcpop,repeat(tests),repeat(td))
    bag = db.from_sequence(xargs, npartitions = npartitions)
    results = list(bag.map(parallel_route).compute())
    pop = [r[0] for r in results]
    dtcpop = [r[1] for r in results]
    '''
    pop,dtcpop = parallel_route(pop,dtcpop,tests,td)
    for ind,d in zip(pop,dtcpop):
        ind.dtc = d
        if not hasattr(ind,'fitness'):
            ind.fitness = copy.copy(pop[0].fitness)
    return pop,dtcpop
    '''
    if isinstance(pop, Iterable) and isinstance(dtcpop,Iterable):
        for p,d in zip(pop,dtcpop):
            if type(p) is type(None) or type(d) is type(None):
                import pdb
                pdb.set_trace()

    #serial route:
    if not isinstance(pop, Iterable):# and not isinstance(dtcpop,Iterable):
        pop,dtcpop = serial_route(pop,td,tests)
        print('serial badness')
    '''

def update_deap_pop(pop, tests, td, backend = None,hc = None):
    '''
    Inputs a population of genes (pop).
    Returned neuronunit scored DataTransportContainers (dtcpop).
    This method converts a population of genes to a population of Data Transport Containers,
    Which act as communicatable data types for storing model attributes.
    Rheobase values are found on the DTCs
    DTCs for which a rheobase value of x (pA)<=0 are filtered out
    DTCs are then scored by neuronunit, using neuronunit models that act in place.
    '''

    #pop = copy.copy(pop)
    if hc is not None:
        pop[0].hc = None
        pop[0].hc = hc

    if backend is not None:
        pop[0].backend = None
        pop[0].backend = backend

    pop, dtcpop = test_runner(pop,td,tests)
    for p,d in zip(pop,dtcpop):
        p.dtc = d
    return pop

def create_subset(nparams = 10, boundary_dict = None):
    # used by GA to find subsets in parameter space.
    if type(boundary_dict) is type(None):
        raise ValueError('A parameter range dictionary was not supplied \
        and the program doesnt know what value to explore.')
        mp = modelp.model_params
        key_list = list(mp.keys())
        reduced_key_list = key_list[0:nparams]
    else:
        key_list = list(boundary_dict.keys())
        reduced_key_list = key_list[0:nparams]
    subset = { k:boundary_dict[k] for k in reduced_key_list }
    return subset
