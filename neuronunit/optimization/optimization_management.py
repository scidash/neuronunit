#import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
#matplotlib.use('agg')

import numpy as np
import dask.bag as db
import pandas as pd
import dask.bag as db
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
import copy
import math
import quantities as pq
import numpy as np
from itertools import repeat
import numpy
from sklearn.cluster import KMeans


from deap import base

from pyneuroml import pynml

from neuronunit.optimization.data_transport_container import DataTC
#from neuronunit.models.interfaces import glif

# Import get_neab has to happen exactly here. It has to be called only on
from neuronunit import tests
from neuronunit.optimization import get_neab
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
from neuronunit.optimization import model_parameters as modelp




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

def inject_and_plot(dtc,figname='problem'):
    '''
    For debugging backends during development.
    '''
    model = mint_generic_model(str('RAW'))
    model.set_attrs(dtc.attrs)
    dtc.vtest[0]['injected_square_current']['amplitude'] = dtc.rheobase['value']
    dtc.vtest[0]['injected_square_current']['duration'] = 1000*pq.ms
    model.inject_square_current(dtc.vtest[0])
    plt.plot(model.get_membrane_potential().times,model.get_membrane_potential())#,label='ground truth')
    plt.savefig(figname+str('debug.png'))

def obtain_predictions(t,backend,params):

    model = None
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
    model.set_attrs(params)
    return t.generate_prediction(model)

def make_fake_observations(tests,backend):
    '''
    to be used in conjunction with round_trip_test below.

    '''
    dtc = DataTC()
    dtc.attrs = local_attrs

    dtc = get_rh(dtc,rtest)

    pt = format_params(tests,rheobase)
    #ptbag = db.from_sequence(pt)
    ptbag = db.from_sequence(pt[1::])

    predictions = list(ptbag.map(obtain_predictions).compute())
    predictions.insert(0,rheobase)

    # having both means and values in dictionary makes it very irritating to iterate over.
    # It's more harmless to demote means to values, than to elevate values to means.
    # Simply swap key names: means, for values.
    for p in predictions:
        if 'mean' in p.keys():
            # here pop means remove from dictionary
            p['value'] = p.pop('mean')

    for ind,t in enumerate(tests):
        if 'mean' in t.observation.keys():
            t.observation['value'] = t.observation.pop('mean')
        pred =  predictions[ind]['value']
        try:
            pred = pred.rescale(t.units)
            t.observation['value'] = pred
        except:
            t.observation['value'] = pred
        t.observation['mean'] = t.observation['value']
    return tests

def round_trip_test(tests,backend):
    '''
    # Inputs:
    #    -- tests, a list of NU test types,
    #    -- backend a string encoding what model, backend, simulator to use.
    # Outputs:
    #    -- a score, that should be close to zero larger is worse.
    # Synopsis:
    #    -- Given any models
    # lets check if the optimizer can find arbitarily sampeled points in
    # a parameter space, using only the information in the error gradient.
    # make some new tests based on internally generated data
    # as opposed to experimental data.
    '''
    explore_param = model_params_everything[backend] # TODO, make this syntax possible.
    noise_param = {} # randomly sample a point in the viable parameter space.
    for t in explore_param.keys():
        mean = np.mean(explore_param[t])
        std = np.std(explore_param[t])
        sample = numpy.random.normal(loc=mean, scale=2*std, size=1)[0]
        noise_param[k] = sample
    tests = make_fake_observations(tests,backend,noise_param)
    NGEN = 10
    MU = 6
    ga_out, DO = run_ga(explore_param,NGEN,tests,free_params=free_params, NSGA = True, MU = MU, backed=backend, selection=str('selNSGA2'))
    best = ga_out['pf'][0].dtc.get_ss()
    print(Bool(best < 0.5))
    if Bool(best >= 0.5:
        NGEN = 10
        MU = 6
        ga_out, DO = run_ga(explore_param,NGEN,tests,free_params=free_params, NSGA = True, MU = MU, backed=backend, selection=str('selNSGA2'),seed_pop=pf[0].dtc.attrs)
        best = ga_out['pf'][0].dtc.get_ss()
    print('Its ',Bool(best < 0.5), ' that optimization succeeds on this model class')
    print('goodness of fit: ',best)

    return Bool(ga_out['pf'][0] < 0.5)

def cluster_tests(use_test,backend,explore_param):
    '''
    Given a group of conflicting NU tests, quickly exploit optimization, and variance information
    To find subsets of tests that don't conflict.
    Inputs:
        backend string, signifying model backend type
        explore_param list of dictionaries, of model parameter limits/ranges.
        use_test, a list of tests per experimental entity.
    Outputs:
        lists of groups of less conflicted test classes.
        lists of groups of less conflicted test class names.
    '''
    MU = 6
    NGEN = 7
    test_opt = {}
    for index,test in enumerate(use_test):
        ga_out, DO = run_ga(explore_param,NGEN,[test],free_params=free_params, NSGA = True, MU = MU,backed=backend, selection=str('selNSGA2'))
        test_opt[test] = ga_out
        with open('qct.p','wb') as f:
            pickle.dump(test_opt,f)

    all_val = {}
    for key,value in test_opt.items():
        all_val[key] = {}
        for k in value['pf'][0].dtc.attrs.keys():
            temp = [i.dtc.attrs[k] for i in value['pf']]
            all_val[key][k] = temp

    first_test = all_val[list(all_val.keys())[0]].values()
    ft = all_val[list(all_val.keys())[0]]
    X = list(first_test)
    X_labels = all_val[list(all_val.keys())[0]].keys()
    df1 = pd.DataFrame(X)
    X = np.array(X)
    est = KMeans(n_clusters=3)
    est.fit(X)
    y_kmeans = est.predict(X)
    first_test = test_opt[list(test_opt.keys())[0]].values()
    test_names = [t.name for t in test_opt.keys()]
    test_classes = [t for t in test_opt.keys()]
    grouped_testsn = {}
    grouped_tests = {}
    for i,k in enumerate(y_kmeans):
        grouped_testsn[k] = []
        grouped_tests[k] = []
    for i,k in enumerate(y_kmeans):
        grouped_testsn[k].append(test_names[i])
        grouped_tests[k].append(test_classes[i])
    return (grouped_tests, grouped_tests)

def mint_generic_model(backend):
    LEMS_MODEL_PATH = path_params['model_path']
    return ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = str(backend))


def write_opt_to_nml(path,param_dict):
    '''
        -- Inputs: desired file path, model parameters to encode in NeuroML2

        -- Outputs: NeuroML2 file.

        -- Synopsis: Write optimimal simulation parameters back to NeuroML2.
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

def pred_only(test_and_models):
    # Temporarily patch sciunit judge code, which seems to be broken.
    #
    #
    (test, dtc) = test_and_models
    obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    pred = test.generate_prediction(model)
    return pred

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
        if hasattr(dtc,'prediction'):# is not None:
            dtc.prediction[test] = pred
            dtc.observation[test] = test.observation['mean']

        else:
            dtc.prediction = None
            dtc.observation = None
            dtc.prediction = {}
            dtc.prediction[test] = pred
            dtc.observation = {}
            dtc.observation[test] = test.observation['mean']


        #dtc.prediction = pred
        score = test.compute_score(obs,pred)
        if not hasattr(dtc,'agreement'):
            dtc.agreement = None
            dtc.agreement = {}
        try:
            dtc.agreement[str(test)] = np.abs(test.observation['mean'] - pred['mean'])
        except:
            try:
                dtc.agreement[str(test)] = np.abs(test.observation['value'] - pred['value'])
            except:
                try:
                    dtc.agreement[str(test)] = np.abs(test.observation['mean'] - pred['value'])
                except:
                    pass
        #print(score.norm_score)
    else:
        score = None
    return score, dtc

def get_rh(dtc,rtest):
    '''
    This is used to generate a rheobase test, given unknown experimental observations.s
    '''
    place_holder = {}
    place_holder['n'] = 86
    place_holder['mean'] = 10*pq.pA
    place_holder['std'] = 10*pq.pA
    place_holder['value'] = 10*pq.pA
    rtest = RheobaseTestP(observation=place_holder,name='a Rheobase test')
    dtc.rheobase = None
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    dtc.rheobase = rtest.generate_prediction(model)
    if dtc.rheobase is None:
        dtc.rheobase = - 1.0
    return dtc


def dtc_to_rheo_serial(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    dtc.scores = {}
    dtc.score = {}
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    rtest = [ t for t in dtc.tests if str('RheobaseTest') == t.name ]
    if len(rtest):
        rtest = rtest[0]
        dtc.rheobase = rtest.generate_prediction(model)
        if dtc.rheobase is not None and dtc.rheobase !=-1.0:
            dtc.rheobase = dtc.rheobase['value']
            obs = rtest.observation
            score = rtest.compute_score(obs,dtc.rheobase)
            dtc.scores[str('RheobaseTest')] = 1.0 - score.norm_score
            if dtc.score is not None:
                dtc = score_proc(dtc,rtest,copy.copy(score))
            rtest.params['injected_square_current']['amplitude'] = dtc.rheobase
        else:
            dtc.rheobase = - 1.0
            dtc.scores[str('RheobaseTest')] = 1.0
    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
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
    # pre format the current injection dictionary based on pre computed
    # rheobase values of current injection.
    # This is much like the hooked method from the old get neab file.
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


def nunit_evaluation_df(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    # same method as below but with data frame.
    tests = dtc.tests
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    LEMS_MODEL_PATH = path_params['model_path']
    df = pd.DataFrame(index=list(tests),columns=['observation','prediction','disagreement'])#,columns=list(reduced_cells.keys()))
    if dtc.rheobase == -1.0 or type(dtc.rheobase) is type(None):
        dtc = allocate_worst(tests,dtc)
    else:
        for k,t in enumerate(tests):
            if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
                t.params = dtc.vtest[k]
                score, dtc= bridge_judge((t,dtc))
                if score is not None:
                    if score.norm_score is not None:
                        dtc.scores[str(t)] = 1.0 - score.norm_score
                        df.iloc[k]['observation'] = t.observation['mean']
                        try:
                            agreement = np.abs(t.observation['mean'] - pred['value'])
                            df.iloc[k]['prediction'] = pred['value']
                            df.iloc[k]['disagreement'] = agreement

                        except:
                            agreement = np.abs(t.observation['mean'] - pred['mean'])
                            df.iloc[k]['prediction'] = pred['mean']
                            df.iloc[k]['disagreement'] = agreement
                else:
                    print('gets to None score type')
    # compute the sum of sciunit score components.
    dtc.summed = dtc.get_ss()
    dtc.df = df
    return dtc

def pred_evaluation(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    tests = dtc.tests
    dtc = copy.copy(dtc)
    # TODO
    # phase out model path:
    # via very reduced model
    dtc.model_path = path_params['model_path']
    preds = []
    dtc.preds = None
    dtc.preds = []
    dtc.preds.append(dtc.rheobase)
    for k,t in enumerate(tests):
        if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
            t.params = dtc.vtest[k]
            pred = pred_only(test_and_models)
            dtc.preds.append(pred)
    return preds

def nunit_evaluation(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    tests = dtc.tests
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    #df = pd.DataFrame(index=list(tests),columns=['observation','prediction','disagreement'])#,columns=list(reduced_cells.keys()))
    if dtc.rheobase == -1.0 or type(dtc.rheobase) is type(None):
        dtc = allocate_worst(tests,dtc)
    else:
        for k,t in enumerate(tests):
            if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
                t.params = dtc.vtest[k]
                score, dtc= bridge_judge((t,dtc))
                if score is not None:
                    if score.norm_score is not None:
                        dtc.scores[str(t)] = 1.0 - score.norm_score

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
        dtcpop[0].ss = None
        dtcpop[0].ss = pop[0].ss
        return dtcpop
    else:
        ind = pop
        for i in ind:
            i.td = td
            i.backend = str(_backend)
        # above replaces need for this line:
        xargs = (ind,td,repeat(backend))
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtc = [ transform(xargs) ]
        dtc.ss = None
        dtc.ss = pop[0].ss
        return dtc




def run_ga(explore_edges, max_ngen, test, free_params = None, hc = None, NSGA = None, MU = None, seed_pop = None, model_type = str('RAW')):
    # seed_pop can be used to
    # to use existing models, that are good guesses at optima, as starting points for optimization.
    # https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python
    # These imports need to be defined with local scope to avoid circular importing problems
    # Try to fix local imports later.

    from neuronunit.optimization.optimisations import SciUnitOptimization


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
    if type(test) is not type([0,0]):
        test = [test]
    DO = SciUnitOptimization(offspring_size = MU, error_criterion = test, boundary_dict = ss, backend = model_type, hc = hc, selection = selection)#,, boundary_dict = ss, elite_size = 2, hc=hc)

    if seed_pop is not None:
        # This is a re-run condition.
        DO.setnparams(nparams = len(free_params), boundary_dict = ss)

        DO.seed_pop = seed_pop
        DO.setup_deap()
    DO.population[0].ss = None
    DO.population[0].ss = ss

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

    optimizer needs a stable
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

    TODO:
    Boot new_genes is a more standard, and less customized implementation of the code below.

    pop, dtcpop = boot_new_genes(number_genes,dtcpop,td)
    dtcpop = map(dtc_to_rheo,dtcpop)

    Whats more boot_new_genes maps many to many (efficient).
    Code below is maps, many to one ( less efficient).

    '''


    impute_gene = [] # impute individual, not impute index
    ind = WSListIndividual()
    for t in td:
        mean = np.mean([ d.attrs[t] for d in dtcpop ])
        std = np.std([ d.attrs[t] for d in dtcpop ])
        sample = numpy.random.normal(loc=mean, scale=2*std, size=1)[0]
        ind.append(sample)
    dtc = DataTC()
    # Brian and PyNN models should not have to read from a file.
    # This line satifies an older NU design flaw, that all models evaluated must have
    # a disk readable path.
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


def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]

def clusty(dtcbag,dtcpop):
    '''
    Used for searching and optimizing where averged double sets of model_parameters
    are the most fundamental gene-unit.
    In other words what is optimized sampled and explored, is the average of two waveform measurements.
    This allows for the ultimate solution, to be expressed as two disparate parameter points, that when averaged
    produce a good model.
    The motivating argument for doing things this way, is because the models, and the experimental data
    results from averaged contributions of measurements from clustered data points making a model with optimal
    error, theoretically unaccessible.
    '''
    from neuronunit.optimization.optimisations import SciUnitOptimization
    # get waveform measurements, and store in genes.
    dtcpop = list(dtcbag.map(pred_evaluation).compute())
    # divide the genes pool in half
    [dtcpopa,dtcpopb] = split_list(copy.copy(dtcpop))
    # average measurements between first half of gene pool, and second half.
    for dtca in dtcpopa:
        for dtcb in dtcpopb:
            dtcb = copy.copy(dtcb)
            for i,x in enumerate(dtcb.preds):
                p = (dtca.preds[i]+dtcb.preds[i])/2
                score = dtca.tests[i].compute_score(obs,p)
                dtca.twin = None
                dtca.twin = dtcb.attrs
                # store the averaged values in the first half of genes.

                dtca.scores[dtca.test.name] = 1.0 - score.norm_score
                score = dtcb.tests[i].compute_score(obs,p)

                #p = (dtca.preds[i]+dtcb.preds[i])/2
                #score = p.compute_score(obs,dtcb.tests[i])
                dtcb.twin = None
                dtcb.twin = dtca.attrs
                # store the averaged values in the second half of genes.

                dtcb.scores[dtcb.test.name] = 1.0 - score.norm_score
    dtcpop = dtcpopa
    dtcpop.extend(dtcpopb)
    return dtcpop


def boot_new_genes(number_genes,dtcpop):
    '''
    Boot strap new genes to make up for completely called onesself.
    '''
    dtcpop = copy.copy(dtcpop)
    DO = SciUnitOptimization(offspring_size = number_genes, error_criterion = [dtcpop[0].test], boundary_dict = boundary, backend = dtcpop[0].backend, selection = str('selNSGA'))#,, boundary_dict = ss, elite_size = 2, hc=hc)
    DO.setnparams(nparams = len(free_params), boundary_dict = ss)
    DO.setup_deap()
    DO.init_pop()
    population = DO.population
    dtcpop = None
    dtcpop = update_dtc_pop(population,DO.td)
    return dtcpop

def parallel_route(pop,dtcpop,tests,td,clustered=False):
    for d in dtcpop:
        d.tests = copy.copy(tests)
    dtcpop = list(map(format_test,dtcpop))
    #import pdb; pdb.set_trace()
    npart = np.min([multiprocessing.cpu_count(),len(dtcpop)])
    dtcbag = db.from_sequence(dtcpop, npartitions = npart)
    if clustered == True:

        dtcpop = list(dtcbag.map(nunit_evaluation).compute())
    else:



    if dtc.score is not None:
        dtc = score_proc(dtc,rtest,copy.copy(score))
    for i,d in enumerate(dtcpop):
        if not hasattr(pop[i],'dtc'):
            pop[i] = WSListIndividual(pop[i])
            pop[i].dtc = None
        d.get_ss()
        pop[i].dtc = copy.copy(d)
    invalid_dtc_not = [ i for i in pop if not hasattr(i,'dtc') ]
    return pop, dtcpop

def make_up_lost(pop,dtcpop,td):
    '''
    make new genes. Two ways to do This
    '''
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

def grid_search(explore_ranges,test_frame,backend=None):
    '''
    Hopefuly this method can depreciate the whole file optimization_management/exhaustive_search.py
    Well actually not quiete. This method does more than that. It iterates over multiple NeuroElectro datum entities.
    A more generalizable method would just act on one NeuroElectro datum entities.
    '''
    store_results = {}
    npoints = 12
    grid = ParameterGrid(explore_ranges)

    size = len(grid)
    temp = []
    if size > npoints:
        sparsify = np.linspace(0,len(grid)-1,npoints)
        for i in sparsify:
            temp.append(grid[int(i)])
        grid = temp
    for local_attrs in grid:
        store_results[str(local_attrs.values())] = {}
        dtc = DataTC()
        #dtc.tests = use_test
        dtc.attrs = local_attrs
        dtc.backend = backend
        dtc.cell_name = backend
        for key, use_test in test_frame.items():
            dtc.tests = use_test
            #dtc = dtc_to_rheo_serial(dtc)
            dtc = dtc_to_rheo(dtc)

            dtc = format_test(dtc)
            if dtc.rheobase is not None:
                if dtc.rheobase!=-1.0:
                    dtc = nunit_evaluation(dtc)
            print(dtc.get_ss())
            store_results[str(local_attrs.values())][key] = dtc.get_ss()
        df = pd.DataFrame(store_results)
        best_params = {}
        for index, row in df.iterrows():
            best_params[index] = row == row.min()
            best_params[index] = best_params[index].to_dict()


        seeds = {}
        for k,v in best_params.items():
            for nested_key,nested_val in v.items():
                if True == nested_val:
                    seed = nested_key
                    seeds[k] = seed
        with open(str(backend)+'_seeds.p','wb') as f:
            pickle.dump(seeds,f)
    return seeds, df


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
    pop,dtcpop = parallel_route(pop,dtcpop,tests,td)
    for ind,d in zip(pop,dtcpop):
        ind.dtc = d
        if not hasattr(ind,'fitness'):
            ind.fitness = copy.copy(pop[0].fitness)
    return pop,dtcpop

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
