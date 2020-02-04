# Its not that this file is responsible for doing plotting,
# but it calls many modules that are, such that it needs to pre-empt
import warnings
import matplotlib
import cython
import logging
try:
    matplotlib.use('agg')
except:
    warnings.warn('X11 plotting backend not available, consider installing')

SILENT = True

if SILENT:
    warnings.filterwarnings("ignore")

PARALLEL_CONFIDENT = True
# Rationale Many methods inside the file optimization_management.py cannot be easily monkey patched using
#```pdb.set_trace()``` unless at the top of the file,
# the parallel_confident static variable is declared false
# This converts parallel mapping functions to serial mapping functions. s
# cheduled Parallel mapping functions cannot tolerate being paused, serial ones can.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import dask.bag as db
import pandas as pd
import pickle
# The rheobase has been obtained seperately and cannot be db mapped.
# Nested DB mappings dont work.
import multiprocessing
npartitions = multiprocessing.cpu_count()
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
import cython



import copy
import math
import quantities as pq
import numpy
try:
    from sciunit import TestSuite
except:
    pass
from neuronunit.optimisation.data_transport_container import DataTC

from neuronunit.optimisation.model_parameters import path_params
from neuronunit.optimisation import model_parameters as modelp
from itertools import repeat
from neuronunit.tests.base import AMPL, DELAY, DURATION
#from neuronunit.models import ReducedModel
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from collections.abc import Iterable
from neuronunit.tests import dm_test_container #import Interoperabe
from neuronunit.tests.base import VmTest

import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# we can explicitly make assignments on it
this.cpucount = multiprocessing.cpu_count()
from neuronunit.tests.fi import RheobaseTestP
from neuronunit.tests.target_spike_current import SpikeCountSearch, SpikeCountRangeSearch
#make_stim_waves = pickle.load(open('waves.p','rb'))

try:
    import asciiplotlib as apl
except:
    warnings.warn('ASCII plot (gnu plot) plotting backend not available, consider installing')
try:
    import efel
except:
    warnings.warn('Blue brain feature extraction not available, consider installing')
try:
    import seaborn as sns
except:
    warnings.warn('Seaborne plotting sub library not available, consider installing')
try:
    from sklearn.cluster import KMeans
except:
    warnings.warn('SKLearn library not available, consider installing')

import time
import seaborn as sns
import random
from datetime import datetime

import pdb
import quantities as pq
from sciunit import scores
from neuronunit.optimisation.optimisations import SciUnitOptimisation
import random
from neuronunit.plottools import elaborate_plots
from neuronunit.plottools import inject_and_plot
# Helper tests are dummy instances of NU tests.
# They are used by other methods analogous to a base class,
# these are base instances that become more derived
# contexts, that modify copies of the helper class in place.
import os
import neuronunit
anchor = neuronunit.__file__
anchor = os.path.dirname(anchor)
mypath = os.path.join(os.sep,anchor,'tests/multicellular_constraints.p')
from neuronunit.optimisation import get_neab
from neuronunit.optimisation.optimisations import WeightedSumFitness

tests = get_neab.process_all_cells()
#import pdb
#pdb.set_trace()
for t in tests.values():
    helper_tests = [value for value in t.tests ]
    break
VERBOSE = False
def timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        logger = logging.getLogger('__main__')
        logging.basicConfig(level=logging.DEBUG)
        if VERBOSE:
            logging.info('Runtime taken to evaluate function {1} {0} seconds'.format(t2-t1,func))
        return f
    return inner

class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def set_fitness(self,obj_size):
        self.fitness = WeightedSumFitness(obj_size=obj_size)
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        if 'dtc' in kwargs:
           self.dtc = kwargs['dtc']
        else:
           self.dtc = None
        super(WSListIndividual, self).__init__(args)#args)#, **kwargs)
        self.obj_size = len(args)
        #self.fitness = tuple(1.0 for i in range(0,self.obj_size))
        self.set_fitness(obj_size=self.obj_size)


def make_ga_DO(explore_edges, max_ngen, test, \
        free_params = None, hc = None,
        selection = None, MU = None, seed_pop = None, \
           backend = str('RAW'),protocol={'allen':False,'elephant':True}):
    '''
    construct an DEAP Optimization Object, suitable for this test class and caching etc.
    '''

    ss = {}
    if type(free_params) is type(dict()):
        if 'dt' in free_params:
            free_params.pop('dt')
        if 'Iext' in free_params:
            free_params.pop('Iext')
    else:
        free_params = [f for f in free_params if str(f) not in 'Iext' and str(f) not in str('dt')]
    for k in free_params:
        if not k in explore_edges.keys() and k not in str('Iext') and k not in str('dt'):

            ss[k] = explore_edges[k]
        else:
            ss[k] = explore_edges[k]
    if type(MU) == type(None):
        MU = 2**len(list(free_params))
    else:
        MU = MU
    max_ngen = int(np.floor(max_ngen))
    if not isinstance(test, Iterable):
        test = [test]
    from neuronunit.optimisation.optimisations import SciUnitOptimisation
    DO = SciUnitOptimisation(MU = MU, error_criterion = test,\
         boundary_dict = ss, backend = backend, hc = hc, \
                             selection = selection,protocol=protocol)

    if seed_pop is not None:
        # This is a re-run condition.
        DO.setnparams(nparams = len(free_params), boundary_dict = ss)

        DO.seed_pop = seed_pop
        DO.setup_deap()
        DO.error_length = len(test)

    return DO

from sciunit import TestSuite

class TSS(TestSuite):
    def __init__(self,tests={},use_rheobase_score=False):
       self.DO = None
       tt = list(tests.values())[0:-1]
       super(TSD,self).__init__(tt)
       #self.update(tests)
       '''
       if 'name' in self.keys():
           self.cell_name = tests['name']
           self.pop('name',None)
       else:
           self.cell_name = 'simulated data'
       '''
       self.use_rheobase_score = use_rheobase_score
       self.elaborate_plots  = elaborate_plots
       self.backend = None


    def optimize(self,param_edges,backend=None,protocol={'allen': False, 'elephant': True},\
        MU=5,NGEN=5,free_params=None,seed_pop=None,hold_constant=None):
        if type(free_params) is type(None):
            free_params=param_edges.keys()
        self.DO = make_ga_DO(param_edges, NGEN, self, free_params=free_params, \
                           backend=backend, MU = 8,  protocol=protocol,seed_pop = seed_pop, hc=hold_constant)
        self.DO.MU = MU
        self.DO.NGEN = NGEN
        ga_out = self.DO.run(NGEN = self.DO.NGEN)
        ga_out['DO'] = self.DO
        if not hasattr(ga_out['pf'][0],'dtc') and 'dtc_pop' not in ga_out.keys():
            _,dtc_pop = self.DO.OM.test_runner(copy.copy(ga_out['pf']),self.DO.OM.td,self.DO.OM.tests)
            ga_out['dtc_pop'] = dtc_pop
        if type(ga_out['pf'][0].dtc) is type(None):
            _,ga_out['dtc_pop'] = self.DO.OM.test_runner(copy.copy(ga_out['pf']),self.DO.OM.td,self.DO.OM.tests)
            # druckman tests can optionally be run on optimized cells here.
            DM = False
            if DM:
                pop,dtcpop = get_dm(ga_out['dtc_pop'],pop=ga_out['pf'])
        else:
            local = [p.dtc for p in ga_out['pf']]
            #ga_out['dtc_pop'] = [ i.dtc for i in ga_out['pf'] ]
            DM = False
            if DM:
                pop,dtcpop = get_dm(local,pop=ga_out['pf'])
            #p in ga_out['pf']],pop=ga_out['pf'])

        self.backend = backend
        '''
        if str(self.cell_name) not in str('simulated data'):
            #pass
            # is this a data driven test? if so its worth plotting results
            ga_out = self.elaborate_plots(self,ga_out)
        '''

        return ga_out, self.DO

class TSD(dict):
    def __init__(self,tests={},use_rheobase_score=False):
       self.DO = None
       self.use_rheobase_score = use_rheobase_score
       self.elaborate_plots  = elaborate_plots
       self.backend = None

      # self.use_rheobase_score = None

       if type(tests) is TestSuite:
           tests = {t.name:t for t in tests.tests}
       if type(tests) is type(dict()):
           pass
       if type(tests) is type(list()):
          tests = {t.name:t for t in tests}

       super(TSD,self).__init__()
       self.update(tests)

       if 'name' in self.keys():
           self.cell_name = tests['name']
           self.pop('name',None)
       else:
           self.cell_name = 'simulated data'


    def optimize(self,param_edges,backend=None,protocol={'allen': False, 'elephant': True},\
        MU=5,NGEN=5,free_params=None,seed_pop=None,hold_constant=None):
        if type(free_params) is type(None):
            free_params=param_edges.keys()
        self.DO = make_ga_DO(param_edges, NGEN, self, free_params=free_params, \
                           backend=backend, MU = 8,  protocol=protocol,seed_pop = seed_pop, hc=hold_constant)
        self.DO.MU = MU
        self.DO.NGEN = NGEN
        ga_out = self.DO.run(NGEN = self.DO.NGEN)
        ga_out['DO'] = self.DO
        if not hasattr(ga_out['pf'][0],'dtc') and 'dtc_pop' not in ga_out.keys():
            _,dtc_pop = self.DO.OM.test_runner(copy.copy(ga_out['pf']),self.DO.OM.td,self.DO.OM.tests)
            ga_out['dtc_pop'] = dtc_pop
        if type(ga_out['pf'][0].dtc) is type(None):
            _,ga_out['dtc_pop'] = self.DO.OM.test_runner(copy.copy(ga_out['pf']),self.DO.OM.td,self.DO.OM.tests)
            DM = False
            if DM:
                pop,dtcpop = get_dm(ga_out['dtc_pop'],pop=ga_out['pf'])
        else:
            local = [p.dtc for p in ga_out['pf']]
            DM = False
            if DM:
                pop,dtcpop = get_dm(local,pop=ga_out['pf'])
        self.backend = backend
        PLOT = False
        if PLOT == True:
            if str(self.cell_name) not in str('simulated data'):
                # is this a data driven test? if so its worth plotting results
                ga_out = self.elaborate_plots(self,ga_out)
        # make ga_out pickleable by cleansing sciunit and deap objects
        for pop in ga_out.values():
            if hasattr(pop,'len'):
                if len(pop):
                    if hasattr(pop[0],'dtc'):
                        for ind in pop:
                            ind.dtc.tests ={ k:v for k,v in ind.dtc.tests.items() }

        return ga_out
# DEAP mutation strategies:
# https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.mutESLogNormal
class WSFloatIndividual(float):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        super(WSFloatIndividual, self).__init__()


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_predictions(dtc):
    if not hasattr(dtc,'preds'):
        dtc.preds = {}
    dtc.preds = {k:v for k,v in dtc.preds.items() if type(v) is not type(int(1))}
    dtc.preds = {k:v for k,v in dtc.preds.items() if type(v['mean']) is not type(None)}
    dtc.preds = {k:v for k,v in dtc.preds.items() if v['mean']!=1.0}
    dtc.preds = {k:v for k,v in dtc.preds.items() if v['mean']!=False}
    dtc.preds = {k:v for k,v in dtc.preds.items() if type(v['mean']) is not type(str(''))}
    dtc.preds = {k:v for k,v in dtc.preds.items() if not np.isnan(v['mean'])}
    return dtc
@cython.boundscheck(False)
@cython.wraparound(False)
def make_new_random(dtc_,backend):
    dtc = DataTC()
    dtc.backend = backend

    dtc.attrs = random_p(backend)
    dtc = dtc_to_rheo(dtc)
    if type(dtc.rheobase) is not type({'1':1}):
        temp = dtc.rheobase
        dtc.rheobase = {'value': temp}
    while dtc.rheobase['value'] is None:
        dtc = DataTC()
        dtc.backend = backend
        dtc.attrs = random_p(backend)
        dtc = dtc_to_rheo(dtc)
        if type(dtc.rheobase) is not type({'1':1}):
            temp = dtc.rheobase
            dtc.rheobase = {'value': temp}
    return dtc



@cython.boundscheck(False)
@cython.wraparound(False)
def random_p(backend):
    ranges = MODEL_PARAMS[backend]
    random_param = {} # randomly sample a point in the viable parameter space.
    for k in ranges.keys():
        try:
            mean = np.mean(ranges[k])
            std = np.std(ranges[k])
            sample = numpy.random.normal(loc=mean, scale=0.25*std, size=1)[0]
            random_param[k] = sample
        except:
            random_param[k] = ranges[k]
    return random_param

@cython.boundscheck(False)
@cython.wraparound(False)
def process_rparam(backend):
    random_param = random_p(backend)
    """
    if 'RAW' in str(backend):
        random_param.pop('Iext',None)
        rp = random_param
        chosen_keys = rp.keys()

    if str('ADEXP') in str(backend):
        random_param.pop('Iext',None)
        rp = random_param
        chosen_keys = rp.keys()
    """
    if 'GLIF' in str(backend):
        random_param['init_AScurrents'] = [0.0,0.0]
        random_param['asc_tau_array'] = [0.3333333333333333,0.01]
        rp = random_param
        chosen_keys = rp.keys()
    else:
        random_param.pop('Iext',None)
        rp = random_param
        chosen_keys = rp.keys()


    dsolution = DataTC()
    dsolution.attrs = rp
    dsolution.backend = backend
    return dsolution,rp,chosen_keys,random_param
def check_test(new_tests):
    replace = False
    for k,t in new_tests.items():
        if type(t.observation['value']) is type(None):
            replace = True
            return replace

def score_only(dtc,pred,test):
    '''

    '''
    if pred is not None:
        if hasattr(dtc,'prediction'):# is not None:
            dtc.prediction[test.name] = pred
            dtc.observation[test.name] = test.observation['mean']

        else:
            dtc.prediction = None
            dtc.observation = None
            dtc.prediction = {test.name: pred}
            dtc.observation = {test.name: test.observation['mean']}

        #dtc.prediction = pred
        score = test.compute_score(obs,pred)
        if not hasattr(dtc,'agreement'):
            dtc.agreement = None
            dtc.agreement = {}
        try:
            dtc.agreement[str(test.name)] = np.abs(test.observation['mean'] - pred['mean'])
        except:
            try:
                dtc.agreement[str(test.name)] = np.abs(test.observation['value'] - pred['value'])
            except:
                try:
                    dtc.agreement[str(test.name)] = np.abs(test.observation['mean'] - pred['value'])
                except:
                    pass
    else:
        score = None
    return score, dtc

# this method is not currently used, but it could also be too prospectively usefull to delete.
# TODO move to a utils file.

def get_centres(ga_out):
    '''
    Do optimization, but then get cluster centres of parameters
    '''
    for key,value in ga_out['pf']:
        all_val[key] = {}
        for k in td.keys():
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
    centers = est.cluster_centers_
    return td, test_opt, centres


def mint_generic_model(backend):
    #this is almost a depricated method
    dtc = DataTC()
    dtc.backend = backend
    model = dtc.to_model()
    return model

import pandas as pd

def save_models_for_justas(dtc):
    with open(str(list(dtc.attrs.values()))+'.csv', 'w') as writeFile:
        df = pd.DataFrame([dtc.attrs])
        writer = csv.writer(writeFile)
        writer.writerows(df)

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
@timer
def pred_only(test_and_models):
    # Temporarily patch sciunit judge code, which seems to be broken.
    #
    #
    (test, dtc) = test_and_models
    #obs = test.observation
    backend_ = dtc.backend
    model = dtc.dtc_to_model()
    #model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    if test.passive:
        test.setup_protocol(model)
        try:
            pred = test.extract_features(model,test.get_result(model))
        except:
            pred = None
    else:
        pred = test.generate_prediction(model)
    return pred

#from functools import partial
#t2m = partial(_pseudo_decor, argument=arg)

#@_pseudo_decor
@timer
def bridge_judge(test_and_dtc):
    (test, dtc) = test_and_dtc
    obs = test.observation
    backend_ = dtc.backend
    model = dtc.dtc_to_model()
    model.set_attrs(dtc.attrs)
    if test.passive:
        test.setup_protocol(model)
        pred = test.extract_features(model,test.get_result(model))
    else:
        pred = test.generate_prediction(model)
    dtc.predictions[test.name] = pred
    dtc.observations[test.name] = test.observation
    if 'mean' in dtc.observations.keys():
        temp = copy.copy(dtc.observations[test.name]['mean'].simplified)
        dtc.observations[test.name]['mean'] = temp
        temp = copy.copy(dtc.predictions[test.name]['mean'].simplified)
    if 'mean' in pred.keys():
        dtc.predictions[test.name]['value'] = pred['mean']

    if type(pred) is not type(None):
        score = test.compute_score(test.observation,pred)

    return score, dtc

def bridge_dm_test(test_and_dtc):
    (test, dtc) = test_and_dtc

    pred_dm = nuunit_dm_rheo_evaluation(dtc)
    if pred_dm.dm_test_features['AP1AmplitudeTest'] is not None:
        width = pred_dm.dm_test_features['AP1WidthHalfHeightTest']
        width_obs = {'mean': width}
        width_obs['std'] = 1.0
        width_obs['n'] = 1

        height = pred_dm.dm_test_features['AP1AmplitudeTest'][0]
        height_obs = {'mean': height}
        height_obs['std'] = 1.0
        height_obs['n'] = 1
        if str('InjectedCurrentAPWidthTest') in test.name:

            score = test.compute_score(test.observation,width_obs)
            return score, dtc

        elif str('InjectedCurrentAPAmplitudeTest') in test.name:

            score = test.compute_score(test.observation,height_obs)
            return score, dtc

from neuronunit.tests import RheobaseTest, RheobaseTestP
def get_rh(dtc,rtest_class):
    '''
    :param dtc:
    :param rtest_class:
    :return:
    This is used to generate a rheobase test, given unknown experimental observations.s
    '''
    place_holder = {'n': 86, 'mean': 10 * pq.pA, 'std': 10 * pq.pA, 'value': 10 * pq.pA}
    backend_ = dtc.backend
    if 'ADEXP' in backend_ or 'GLIF' in backend_ or 'BHH' in backend_:
        rtest = RheobaseTest(observation=place_holder,
                                name='RheobaseTest')

        #rtest = RheobaseTestP(observation=place_holder,
        #                        name='RheobaseTest')
    else:
        rtest = RheobaseTest(observation=place_holder,
                         name='RheobaseTest')
    dtc.rheobase = None
    model = dtc.dtc_to_model()
    model.set_attrs(dtc.attrs)
    rtest.params['injected_square_current'] = {}
    rtest.params['injected_square_current']['delay'] = DELAY
    rtest.params['injected_square_current']['duration'] = DURATION
    dtc.rheobase = rtest.generate_prediction(model)
    if dtc.rheobase is None:
        dtc.rheobase = - 1.0
    return dtc


def substitute_parallel_for_serial(rtest):
    rtest = RheobaseTestP(rtest.observation)
    return rtest

def is_parallel_rheobase_compatible(backend):
    incompatible_backends = ['RAW', 'HH']
    incompatible = any([x in backend for x in incompatible_backends])
    return not incompatible

def get_new_rtest(dtc):
    place_holder = {'n': 86, 'mean': 10 * pq.pA, 'std': 10 * pq.pA, 'value': 10 * pq.pA}
    f = RheobaseTestP if is_parallel_rheobase_compatible(dtc.backend) else RheobaseTest
    return f(observation=place_holder, name='RheobaseTest')

def get_rtest(dtc):
    if not hasattr(dtc, 'tests'):
        rtest = get_new_rtest(dtc)
    else:
        if type(dtc.tests) is type(list()):
           rtests = [t for t in dtc.tests if 'rheo' in t.name.lower()]
        else:
           rtests = [v for k,v in dtc.tests.items() if 'rheo' in str(k).lower()]

        if len(rtests):
            rtest = rtests[0]
            if is_parallel_rheobase_compatible(dtc.backend):
                rtest = substitute_parallel_for_serial(rtest)
        else:
            rtest = get_new_rtest(dtc)
    return rtest


def dtc_to_model(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    if not hasattr(dtc,'scores'):
        dtc.scores = None
    if type(dtc.scores) is type(None):
        dtc.scores = {}
    model = dtc.dtc_to_model()
    model.attrs = dtc.attrs
    return model

def dtc_to_rheo(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    if not hasattr(dtc,'scores'):
        dtc.scores = None
    if type(dtc.scores) is type(None):
        dtc.scores = {}
    model = dtc.dtc_to_model()

    if hasattr(dtc,'tests'):
        if type(dtc.tests) is type({}) and str('RheobaseTest') in dtc.tests.keys():
            rtest = dtc.tests['RheobaseTest']
            if str('JHH') in dtc.backend or str('BHH') in dtc.backend or str('ADEXP') in dtc.backend or str("GLIF") in dtc.backend:
                 rtest = RheobaseTestP(dtc.tests['RheobaseTest'].observation)
        else:
            rtest = get_rtest(dtc)
    else:
        rtest = get_rtest(dtc)

    if rtest is not None:
        if isinstance(rtest,Iterable):
            rtest = rtest[0]
        dtc.rheobase = rtest.generate_prediction(model)['value']
        return dtc

        if dtc.rheobase is not None:
            if type(dtc.rheobase['value']) is not type(None):
                if not hasattr(dtc,'prediction'):
                    dtc.prediction = {}
                dtc.prediction[str(rtest.name)] = dtc.rheobase
                dtc.rheobase = dtc.rheobase['value']
                rtest.prediction = dtc.rheobase
                return dtc

    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
    return dtc

def inject_and_plot_passive_model(attrs,backend):
    pre_model = DataTC()
    pre_model.attrs = attrs
    pre_model.backend = backend
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    model = pre_model.dtc_to_model()
    DURATION = 500.0*pq.ms
    DELAY = 200.0*pq.m
    uc = {'amplitude':-10*pq.pA,'duration':DURATION,'delay':DELAY}
    model.inject_square_current(uc)
    vm = model.get_membrane_potential()
    plt.plot(vm.times,vm.magnitude)
    plt.show()
    return vm,plt

def inject_and_plot_model(attrs,backend):
    pre_model = DataTC()
    pre_model.attrs = attrs
    pre_model.backend = backend
    # get rheobase injection value
    pre_model = dtc_to_rheo(pre_model)
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    model = pre_model.dtc_to_model()
    uc = {'amplitude':model.rheobase,'duration':DURATION,'delay':DELAY}

    model.inject_square_current(uc)
    vm = model.get_membrane_potential()
    plt.plot(vm.times,vm.magnitude)
    plt.show()
    return vm,plt


def score_proc(dtc,t,score):
    dtc.score[str(t)] = {}
    if hasattr(score,'norm_score'):
        dtc.score[str(t)]['value'] = copy.copy(score.log_norm_score)
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

def switch_logic(xtests):
    # move this logic into sciunit tests
    '''
    Hopefuly depreciated by future NU debugging.
    '''
    aTSD = neuronunit.optimisation.optimization_management.TSD()
    if type(xtests) is type(aTSD):
        xtests = list(xtests.values())
    if type(xtests) is type(list()):
        pass
    for t in xtests:
        if str('RheobaseTest') == t.name:
            t.active = True
            t.passive = False
        elif str('RheobaseTestP') == t.name:
            t.active = True
            t.passive = False
        elif str('InjectedCurrentAPWidthTest') == t.name:
            t.active = True
            t.passive = False
        elif str('InjectedCurrentAPAmplitudeTest') == t.name:
            t.active = True
            t.passive = False
        elif str('InjectedCurrentAPThresholdTest') == t.name:
            t.active = True
            t.passive = False
        elif str('RestingPotentialTest') == t.name:
            t.passive = True
            t.active = False
        elif str('InputResistanceTest') == t.name:
            t.passive = True
            t.active = False
        elif str('TimeConstantTest') == t.name:
            t.passive = True
            t.active = False
        elif str('CapacitanceTest') == t.name:
            t.passive = True
            t.active = False
        else:
            t.passive = False
            t.active = False
    return xtests

def active_values(keyed,rheobase,square = None):
    keyed['injected_square_current'] = {}
    if square is None:
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase

        keyed['injected_square_current']['delay'] = DELAY
        keyed['injected_square_current']['duration'] = DURATION

    else:
        keyed['injected_square_current']['duration'] = square['Time_End'] - square['Time_Start']
        keyed['injected_square_current']['delay'] = square['Time_Start']
        keyed['injected_square_current']['amplitude'] = square['prediction']#value'])*pq.pA

    return keyed

def passive_values(keyed):
    PASSIVE_DURATION = 500.0*pq.ms
    PASSIVE_DELAY = 200.0*pq.ms
    keyed['injected_square_current'] = {}
    keyed['injected_square_current']['delay']= PASSIVE_DELAY
    keyed['injected_square_current']['duration'] = PASSIVE_DURATION
    keyed['injected_square_current']['amplitude'] = -10*pq.pA
    return keyed




def allocate_worst(xtests,dtc):
    # If the model fails tests, and cannot produce model driven data
    # Allocate the worst score available.
    if not hasattr(dtc,'scores'):
        dtc.scores = {}
    if type(dtc.scores) is type(None):
        dtc.scores = {}
    for t in xtests:
        dtc.scores[str(t.name)] = 1.0

    return dtc




def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))

import copy

def get_dtc_pop(contains_dtcpop,filtered_tests,model_parameters,backend = 'ADEXP'):
    from neuronunit.optimisation.optimisations import SciUnitOptimisation

    random.seed(64)
    boundary_dict = model_parameters.MODEL_PARAMS[backend]
    tests=filtered_tests['Hippocampus CA1 basket cell']
    try:
        DO = SciUnitOptimisation(offspring_size = 1,
            error_criterion = tests, boundary_dict = boundary_dict,
                                     backend = backend, selection = str('selNSGA'))#,simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)
    except:
        DO = SciUnitOptimisation(offspring_size = 1,
            error_criterion = tests, boundary_dict = boundary_dict,
                                     backend = backend, selection = str('selNSGA'))#,simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)

    DO.setnparams(nparams = len(contains), boundary_dict = boundary_dict)
    DO.setup_deap()

    dtcdic = {}
    for k,v in contains_dtcpop[backend].items():

        dtcpop = []
        for i in v:
            dtcpop.append(transform((i,DO.td,backend)))
            dtcpop[-1].backend = backend
            dtcpop[-1] = DO.OptMan.dtc_to_rheo(dtcpop[-1])
            dtcpop[-1] = DO.OptMan.format_test(dtcpop[-1])

        dtcdic[k] = copy.copy(dtcpop)
    return dtcdic, DO

def allen_wave_predictions(vm30):
    dtc = DataTC()
    try:
        vm30.rescale(pq.V)
    except:
        pass
    v = [float(v*1000.0) for v in vm30.magnitude]
    t = [float(t) for t in vm30.times]
    try:
        #spks = ft.detect_putative_spikes(np.array(v),np.array(t))
        ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
        ephys.process_spikes()

    except:
        '''
        rectify unfilterable high sample frequencies by downsampling them
        downsample too densely sampled signals.
        Making them amenable to Allen analysis
        '''

        #if dtc.backend in str('ADEXP'):
        #    vm30 = model.finalize()
        v = [ float(v*1000.0) for v in vm30.magnitude]
        t = [ float(t) for t in vm30.times ]
        ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
        ephys.process_spikes()

    ephys_dict = ephys.as_dict()
    if not 'spikes' in ephys_dict.keys() or ephys_dict['avg_rate'] == 0:
        dtc.scores = None
        dtc.preds = {}
        return dtc
    else:
        prediction = {}
        dtc.preds= {}
        obs= {}
        for k in ephys_dict.keys():
            if 'spikes' not in k:
                prediction['mean'] = ephys_dict[k]
                prediction['std'] = 10.0
                dtc.preds[k] = prediction

            else:

                for wavef in ephys_dict['spikes'][0].keys():
                    temp = ephys_dict['spikes'][0][wavef]
                    prediction['mean'] = temp
                    prediction['std'] = 10.0
                    dtc.preds[wavef+str('_first')] = prediction
                for wavef in ephys_dict['spikes'][-1].keys():
                    temp = ephys_dict['spikes'][-1][wavef]
                    prediction['mean'] = temp
                    prediction['std'] = 10.0
                    dtc.preds[wavef+str('_last')] = prediction
                half = int(len(ephys_dict['spikes'])/2.0)
                for wavef in ephys_dict['spikes'][half].keys():
                    temp = ephys_dict['spikes'][half][wavef]
                    prediction['mean'] = temp
                    prediction['std'] = 10.0
                    dtc.preds[wavef+str('_half')] = prediction

                dtc.spike_cnt = len(ephys_dict['spikes'])
                dtc.preds['spikes'] = dtc.spike_cnt

        return dtc,ephys
from scipy.interpolate import interp1d

# from scipy.interpolate import interp1d

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def append_spikes(ephys_dict,dtc):
    prediction = {}
    dtc.preds= {}
    obs= {}
    for k in ephys_dict.keys():
        if 'spikes' not in k:
            prediction['mean'] = ephys_dict[k]
            prediction['std'] = 10.0
            dtc.preds[k] = prediction

        else:

            for wavef in ephys_dict['spikes'][0].keys():
                temp = ephys_dict['spikes'][0][wavef]
                prediction['mean'] = temp
                prediction['std'] = 10.0
                dtc.preds[wavef+str('_first')] = prediction
            for wavef in ephys_dict['spikes'][-1].keys():
                temp = ephys_dict['spikes'][-1][wavef]
                prediction['mean'] = temp
                prediction['std'] = 10.0
                dtc.preds[wavef+str('_last')] = prediction
            half = int(len(ephys_dict['spikes'])/2.0)
            for wavef in ephys_dict['spikes'][half].keys():
                temp = ephys_dict['spikes'][half][wavef]
                prediction['mean'] = temp
                prediction['std'] = 10.0
                dtc.preds[wavef+str('_half')] = prediction

            '''
            Fit all the spikes
            for i in ephys_dict['spikes']:
                for wavef in i.keys():
                    temp = i[wavef]
                    prediction['mean'] = temp
                    prediction['std'] = 10.0
                    dtc.preds[wavef+str(i) = prediction
            '''
        dtc.spike_cnt = len(ephys_dict['spikes'])
        dtc.preds['spikes'] = dtc.spike_cnt
    return dtc

from neuronunit.tests.target_spike_current import SpikeCountSearch, SpikeCountRangeSearch

def just_allen_predictions(dtc):
    if type(dtc.ampl) is not type(None):
        current = {'injected_square_current':
                    {'amplitude':dtc.ampl, 'delay':DELAY, 'duration':DURATION}}
    else:
        if 'spike_count' in dtc.preds.keys():
            observation_spike = {'vale': dtc.preds['spike_count']['mean']} # -1,dtc.preds['spike_count']['mean']+1] }
            scs = SpikeCountSearch(observation_spike)
            model = new_model(dtc)
            dtc.ampl = scs.generate_prediction(model)
        else:
            dtc = dtc_to_rheo(dtc)
            dtc.ampl = 3.0 * dtc.rheobase#['value']
    current = {'injected_square_current':
               {'amplitude':dtc.ampl, 'delay':DELAY, 'duration':DURATION}}

    comp = False
    if hasattr(dtc,'pre_obs'):

        if type(dtc.pre_obs) is not type(None):
            compare = dtc.pre_obs
            comp = True
            if 'spk_count' not in compare.keys():
                # TODO unalias these variables.
                target = compare['spike_count']
            else:
                target = compare['spk_count']
        else:
            compare = None

    else:
        compare = None

    model = new_model(dtc)
    assert model is not None
    vm30 = model.inject_square_current(current['injected_square_current'])
    vm30 = model.get_membrane_potential()
    dtc.current = current
    if comp:
        if model.get_spike_count()!=target:

            dtc.scores = None
            dtc.fitted_spike_cnt = False
        else:
            dtc.fitted_spike_cnt = True
    if np.max(vm30)<0.0 or model.get_spike_count()<1:
        dtc.scores = None
        return dtc
    model.vm30 = None
    model.vm30 = vm30
    try:
        vm30.rescale(pq.V)
    except:
        pass
    v = [float(v*1000.0) for v in vm30.magnitude]
    t = [float(t) for t in vm30.times]
    try:
        #spks = ft.detect_putative_spikes(np.array(v),np.array(t))
        ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
        ephys.process_spikes()

    except:
        '''
        rectify unfilterable high sample frequencies by downsampling them
        downsample too densely sampled signals.
        Making them amenable to Allen analysis
        '''

        if dtc.backend in str('ADEXP'):
            vm30 = model.finalize()
            v = [ float(v*1000.0) for v in vm30.magnitude]
            t = [ float(t) for t in vm30.times ]
        try:
            ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
            ephys.process_spikes()
        except:
            return dtc
    ephys_dict = ephys.as_dict()
    if not 'spikes' in ephys_dict.keys() or ephys_dict['avg_rate'] == 0:
        dtc.scores = None
        dtc.preds = {}
        return dtc
    else:
        def use_later(npts):
            '''
            garuntee a maximum number of aligned reference indexs in disparate length spike trains.
            '''
            garunteed_reference_points = downsample(list(range(0,len(ephys_dict['spikes'])), npts))
        dtc = append_spikes(ephys_dict,dtc)
        return dtc,compare,ephys


from sciunit import scores

def prediction_current_and_features(dtc):
    returned = just_allen_predictions(dtc)
    try:
        dtc,compare,ephys = returned
    except:
        dtc = returned
        return dtc

    ephys_dict = ephys.as_dict()
    dtc.scores = None
    dtc.scores = {}
    for k,v in compare.items():
        dtc.scores[k] = 1.0
    helper = helper[0]
    #score_type = scores.RatioScore
    score_type = scores.ZScore
    #helper.score_type = score_type

    helper.score_type = score_type

    dtc.preds = {}
    dtc.tests = {}

    for k,observation in compare.items():
        if  str('spikes') not in k:
            '''
            compute interspike firing frame_dynamics
            '''

            obs = {}
            if type(observation) is not type({'1':0}):
                obs['mean'] = observation
            else:
                obs = observation
            prediction = {}
            if k in ephys_dict.keys():
                prediction['mean'] = ephys_dict[k]
                helper.name = str(k)
                obs['std'] = obs['mean']/10.0
                dtc.preds[k] = prediction
                prediction['std'] = prediction['mean']/10.0

                try:
                    score = VmTest.compute_score(helper,obs,prediction)
                except:
                    score = None
                dtc.tests[k] = VmTest(obs)#.compute_score(helper,obs,prediction)
                if score is not None and score.log_norm_score is not None:
                    dtc.scores[k] = 1.0-float(score.log_norm_score)
                else:
                    dtc.scores[k] = 1.0

        '''
        compute perspike waveform features on just the first spike
        '''
        first_spike = ephys_dict['spikes'][0]
        half_spike = ephys_dict['spikes'][int(len(ephys_dict['spikes'])/2.0)]
        last_spike = ephys_dict['spikes'][-1]

        first_spike.pop('direct',None)
        temp = ['_first','_half','_last']
        for i,spike in enumerate([first_spike,half_spike,last_spike]):
            for key,spike_obs in spike.items():
                if i == 0:
                    obs = {'mean': compare[key+str('_first')]['mean']}
                elif i == 1:
                    obs = {'mean': compare[key+str('_half')]['mean']}
                elif i == 2:
                    obs = {'mean': compare[key+str('_last')]['mean']}

                #if not str('direct') in key and not str('adp_i') in key and not str('peak_i') in key and not str('fast_trough_i') and not str('fast_trough_i') and not str('trough_i'):
                #try:


                prediction = {'mean': ephys_dict['spikes'][0][key]}
                helper.name = str(key)
                #obs['std']=10.0
                #prediction['std']=10.0
                #dtc.preds[key] = prediction
                obs['std'] = obs['mean']/15.0
                dtc.preds[k] = prediction
                if type(prediction['mean']) is not type(str()):
                    prediction['std'] = prediction['mean']/15.0
                    score = VmTest.compute_score(helper,obs,prediction)
                else:
                    score = None
                dtc.tests[key] = VmTest(obs)
                if not score is None and not score.log_norm_score is None:
                    dtc.scores[key] = 1.0-score.log_norm_score
                else:
                    dtc.scores[key] = 1.0
    dtc.ephys = None
    dtc.ephys = ephys
    dtc.spike_number = len(ephys_dict['spikes'])
    try:
        dtc.scores['spike_count'] = np.abs(spike_cnt_pred - target)# sigmoid(np.abs(delta))
    except:
        pass
    try:
        delta = np.abs(float(dtc.ampl)-dtc.pre_obs['current_test'])
        dtc.scores['current_test'] = delta# sigmoid(np.abs(delta))
    except:
        pass

    return dtc

def new_model(dtc):
    model = dtc.dtc_to_model()
    #model = mint_generic_model(dtc.backend)
    model.set_attrs(dtc.attrs)
    return model

def make_stim_waves_func():
    import allensdk.core.json_utilities as json_utilities
    import pickle
    neuronal_model_id = 566302806
    # download model metadata
    try:
        ephys_sweeps = json_utilities.read('ephys_sweeps.json')
    except:
        from allensdk.api.queries.glif_api import GlifApi

        glif_api = GlifApi()
        nm = glif_api.get_neuronal_models_by_id([neuronal_model_id])[0]

        from allensdk.core.cell_types_cache import CellTypesCache
        # download information about the cell
        ctc = CellTypesCache()
        ctc.get_ephys_data(nm['specimen_id'], file_name='stimulus.nwb')
        ctc.get_ephys_sweeps(nm['specimen_id'], file_name='ephys_sweeps.json')
        ephys_sweeps = json_utilities.read('ephys_sweeps.json')

    ephys_file_name = 'stimulus.nwb'


    sweep_numbers = [ s['sweep_number'] for s in ephys_sweeps if s['stimulus_units'] == 'Amps' ]

    #snumber = [ s for s in ephys_sweeps if s['stimulus_units'] == 'Amps' if s['num_spikes']>=1]
    stimulus = [ s for s in ephys_sweeps if s['stimulus_units'] == 'Amps' \
     if s['num_spikes'] != None \
     if s['stimulus_name']!='Ramp' and s['stimulus_name']!='Short Square']

    amplitudes = [ s['stimulus_absolute_amplitude'] for s in stimulus ]
    durations = [ s['stimulus_duration'] for s in stimulus ]

    expeceted_spikes = [ s['num_spikes'] for s in stimulus ]
    #durations = [ s['stimulus_absolute_amplitude'] for s in stimulus ]
    delays = [ s['stimulus_start_time'] for s in stimulus ]
    sn = [ s['sweep_number'] for s in stimulus ]
    make_stim_waves = {}
    for i,j in enumerate(sn):
        make_stim_waves[j] = {}
        make_stim_waves[j]['amplitude'] = amplitudes[i]
        make_stim_waves[j]['delay'] = delays[i]
        make_stim_waves[j]['durations'] = durations[i]
        make_stim_waves[j]['expeceted_spikes'] = expeceted_spikes[i]

    pickle.dump(make_stim_waves,open('waves.p','wb'))
    return make_stim_waves





def nuunit_allen_evaluation(dtc):

    if hasattr(dtc,'vtest'):
        values = [v for v in dtc.vtest.values()]
        for value in values:
            if str('injected_square_current') in value.keys():
                current = value['injected_square_current']
                current['amplitude'] = dtc.rheobase * 3.0
                break
    else:
        try:
            observation_spike = {'range': dtc.tsr}
        except:
            dtc.tsr = [ dtc.pre_obs['spike_count']['mean']-2, dtc.pre_obs['spike_count']['mean']+5]
            observation_spike = {'range': dtc.tsr}

        if not str('GLIF') in str(dtc.backend):
            scs = SpikeCountRangeSearch(observation_spike)
            model = new_model(dtc)
            assert model is not None
            target_current = scs.generate_prediction(model)

            dtc.ampl = None
            if target_current is not None:
                dtc.ampl = target_current['value']
                dtc = prediction_current_and_features(dtc)
                dtc = filter_predictions(dtc)
                dtc.error_length = len(dtc.preds)
            if target_current is None:
                dtc.ampl = None
                dtc.preds = {}
                return dtc
        else:
            try:
                with open('waves.p','wb') as f:
                    make_stim_waves = pickle.load(f)
            except:
                make_stim_waves = make_stim_waves_func()
            dtc = dtc_to_rheo(dtc)
            dtc.ampl = dtc.rheobase['value']*3.0
            dtc = prediction_current_and_features(dtc)
            dtc = filter_predictions(dtc)

            dtc.error_length = len(dtc.preds)

        return dtc


def nuunit_dm_evaluation(dtc):
    model = dtc.dtc_to_model()
    model.set_attrs(dtc.attrs)
    try:
        values = [v for v in dtc.protocols.values()][0]

    except:
        values = [v for v in dtc.tests.values()][0]
    current = values['injected_square_current']

    current['amplitude'] = dtc.rheobase * 1.5
    model.inject_square_current(current)
    vm15 = model.get_membrane_potential()

    model.vm15 = None
    model.vm15 = vm15
    model.druckmann2013_standard_current = None
    model.druckmann2013_standard_current = dtc.rheobase * 1.5
    current['amplitude'] = dtc.rheobase * 3.0

    vm30 = model.inject_square_current(current)
    vm30 = model.get_membrane_potential()
    if dtc.rheobase <0.0 or np.max(vm30)<0.0 or model.get_spike_count()<1:
        dtc.dm_test_features = None
        return dtc
    model.vm30 = None
    model.vm30 = vm30
    model.druckmann2013_strong_current = None
    model.druckmann2013_strong_current = dtc.rheobase * 3.0

    model.druckmann2013_input_resistance_currents =[ -5.0*pq.pA, -10.0*pq.pA, -15.0*pq.pA]#,copy.copy(current)

    DMTNMLO = dm_test_container.DMTNMLO()
    DMTNMLO.test_setup(None,None,model= model)
    dm_test_features = DMTNMLO.runTest()
    dtc.AP1DelayMeanTest = None
    dtc.AP1DelayMeanTest = dm_test_features['AP1DelayMeanTest']
    dtc.InputResistanceTest = dm_test_features['InputResistanceTest']

    dtc.dm_test_features = None
    dtc.dm_test_features = dm_test_features
    return dtc


def input_resistance_dm_evaluation(dtc):
    model = dtc.dtc_to_model()
    #model = mint_generic_model(dtc.backend)
    model.set_attrs(dtc.attrs)
    try:
        values = [v for v in dtc.protocols.values()][0]

    except:
        values = [v for v in dtc.tests.values()][0]

        #values = [v for v in dtc.vtest.values()][0]

    current = values['injected_square_current']

    current['amplitude'] = dtc.rheobase * 1.5
    model.inject_square_current(current)
    vm15 = model.get_membrane_potential()

    model.vm15 = None
    model.vm15 = vm15
    model.druckmann2013_standard_current = None
    model.druckmann2013_standard_current = dtc.rheobase * 1.5
    current['amplitude'] = dtc.rheobase * 3.0

    vm30 = model.inject_square_current(current)
    vm30 = model.get_membrane_potential()
    if dtc.rheobase <0.0 or np.max(vm30)<0.0 or model.get_spike_count()<1:
        dtc.dm_test_features = None
        return dtc
    model.vm30 = None
    model.vm30 = vm30
    model.druckmann2013_strong_current = None
    model.druckmann2013_strong_current = dtc.rheobase * 3.0

    model.druckmann2013_input_resistance_currents =[ -5.0*pq.pA, -10.0*pq.pA, -15.0*pq.pA]#,copy.copy(current)

    DMTNMLO = dm_test_container.DMTNMLO()
    DMTNMLO.test_setup(None,None,model= model)
    dm_test_features = DMTNMLO.runTest()
    dtc.AP1DelayMeanTest = None
    dtc.AP1DelayMeanTest = dm_test_features['AP1DelayMeanTest']
    dtc.InputResistanceTest = dm_test_features['InputResistanceTest']

    dtc.dm_test_features = None
    dtc.dm_test_features = dm_test_features
    return dtc

def nuunit_dm_rheo_evaluation(dtc):
    model = dtc.dtc_to_model()
    #model = mint_generic_model(dtc.backend)
    model.set_attrs(dtc.attrs)
    #values = [v for v in dtc.vtest.values()][0]

    try:
        values = [v for v in dtc.protocols.values()][0]
    except:
        values = [v for v in dtc.tests.values()][0]

    current = values['injected_square_current']

    current['amplitude'] = dtc.rheobase
    model.inject_square_current(current)
    vm15 = model.get_membrane_potential()

    model.vm15 = None
    model.vm15 = vm15
    model.druckmann2013_standard_current = None
    model.druckmann2013_standard_current = dtc.rheobase * 1.5
    current['amplitude'] = dtc.rheobase * 3.0

    vm30 = model.inject_square_current(current)
    vm30 = model.get_membrane_potential()
    if dtc.rheobase <0.0 or np.max(vm30)<0.0 or model.get_spike_count()<1:
        return dtc
    model.vm30 = None
    model.vm30 = vm30
    model.druckmann2013_strong_current = None
    model.druckmann2013_strong_current = dtc.rheobase * 3.0

    model.druckmann2013_input_resistance_currents =[ -5.0*pq.pA, -10.0*pq.pA, -15.0*pq.pA]#,copy.copy(current)

    DMTNMLO = dm_test_container.DMTNMLO()
    DMTNMLO.test_setup_subset(None,None,model= model)
    dm_test_features = DMTNMLO.runTest()

    dtc.dm_test_features = None
    dtc.dm_test_features = dm_test_features
    return dtc


from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor


def efel_evaluation(dtc):

    try:
        fig = apl.figure()
        fig.plot(t, v, label=str('spikes: '), width=100, height=20)
        fig.show()
    except:
        pass
    v = [float(v)*1000.0 for v in vm30.magnitude]
    t = [float(t) for t in vm30.times]

    ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
    ephys.process_spikes()
    spks = ft.detect_putative_spikes(np.array(v),np.array(t))
    dtc.spks = None
    dtc.spks = spks
    dtc.ephys = None
    dtc.ephys = ephys
    return dtc
    # possibly depricated
    ##
    # Wrangle data to prepare for EFEL feature calculation.
    ##
    values = [v for v in dtc.vtest.values()][0]
    current = values['injected_square_current']

    current['amplidtude'] = dtc.rheobase * 1.5
    model.protocol = None
    model.protocol = {'Time_End': current['duration'] + current['delay'], 'Time_Start': current['delay']}

    trace3 = {'T': [float(t) for t in model.vm30.times.rescale('ms')], 'V': [float(v) for v in model.vm30.magnitude],
              'stimulus_current': [model.druckmann2013_strong_current]}
    trace3['stim_end'] = [ trace3['T'][-1] ]
    trace3['stim_start'] = [ trace3['T'][0] ]
    traces3 = [trace3]# Now we pass 'traces' to the efel and ask it to calculate the feature# values

    trace15 = {}
    trace15['T'] = [float(t) for t in model.vm15.times.rescale('ms')]
    trace15['V'] = [float(v) for v in model.vm15.magnitude]
    trace15['stim_start'] = [ trace15['T'][0] ]
    trace15['stimulus_current'] = [ model.druckmann2013_standard_current ]
    trace15['stim_end'] = [ trace15['T'][-1] ]
    traces15 = [trace15]# Now we pass 'traces' to the efel and ask it to calculate the feature# values

    ##
    # Compute
    # EFEL features (HBP)
    ##
    efel.reset()

    if len(threshold_detection(model.vm15, threshold=0)):
        threshold = float(np.max(model.vm15.magnitude)-0.5*np.abs(np.std(model.vm15.magnitude)))


    #efel_15 = efel.getMeanFeatureValues(traces15,list(efel.getFeatureNames()))#
    else:
        threshold = float(np.max(model.vm15.magnitude)-0.2*np.abs(np.std(model.vm15.magnitude)))
    efel.setThreshold(threshold)
    if np.min(model.vm15.magnitude)<0:
        efel_15 = efel.getMeanFeatureValues(traces15,list(efel.getFeatureNames()))
    else:
        efel_15 = None
    efel.reset()
    if len(threshold_detection(model.vm30, threshold=0)):
        threshold = float(np.max(model.vm30.magnitude)-0.5*np.abs(np.std(model.vm30.magnitude)))
    else:
        threshold = float(np.max(model.vm30.magnitude)-0.2*np.abs(np.std(model.vm30.magnitude)))
        efel.setThreshold(threshold)
    if np.min(model.vm30.magnitude)<0:
        efel_30 = efel.getMeanFeatureValues(traces3,list(efel.getFeatureNames()))
    else:
        efel_30 = None
    efel.reset()
    dtc.efel_15 = None
    dtc.efel_30 = None
    dtc.efel_15 = efel_15
    dtc.efel_30 = efel_30



def dtc_to_predictions(dtc):
    dtc.preds = {}
    for t in dtc.tests:
        preds = t.generate_prediction(dtc.dtc_to_model())
        dtc.preds[t.name] = preds
    return dtc

def evaluate_allen(dtc,regularization=True):
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    fitness = [ 1.0 for i in range(0,len(dtc.ascores)) ]
    if regularization:
        fitness0 = (t**(1.0/2.0) for int_,t in enumerate(dtc.ascores.keys()))
        fitness1 = (np.abs(t) for int_,t in enumerate(dtc.ascores.keys()))
        fitness = [ (fitness1[i]+j)/2.0 for i,j in enumerate(fitness0)]
        return tuple(fitness,)
    else:
        if dtc.ascores[str(t)] is None:
            fitness[int_] = 1.0
        else:
            fitness[int_] = dtc.ascores[str(t)]
    return tuple(fitness,)

def evaluate_sm(dtc,regularization=False,elastic_net=False):
    # assign worst case errors, and then over write them with situation informed errors as they become available.

    if not hasattr(dtc,str('SM')):
        return fitness
    fitness = []
    for key,value in zip(dtc.SM.keys(),dtc.SM.values()):
        fitness.append(float(value))
    return tuple(fitness,)

def evaluate(dtc,regularization=False,elastic_net=False):
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    if not hasattr(dtc,str('SM')):
        return []
    else:

        fitness = tuple(dtc.SM.score.values[0],)
        return fitness

def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list

from sciunit import scores


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

def filtered(pop,dtcpop):
    dtcpop = [ dtc for dtc in dtcpop if type(dtc.rheobase) is not type(None) ]
    pop = [ p for p in pop if type(p.rheobase) is not type(None) ]
    if len(pop) != len(dtcpop):
        print('fatal')
    assert len(pop) == len(dtcpop)
    return pop, dtcpop


def split_list(a_list):
    half = int(len(a_list)/2)
    return a_list[:half], a_list[half:]


def which_thing(thing):
    if 'value' in thing.keys():
        standard = thing['value']
    if 'mean' in thing.keys():
        standard = thing['mean']
    thing['standard'] = standard
    return thing

def which_key(thing):

    if 'value' in thing.keys():
        return 'value'
    if 'mean' in thing.keys():
        return 'mean'

def simple_error(observation,prediction,t):
    observation = which_thing(observation)
    prediction = which_thing(prediction)
    obs = observation['standard']
    pred = prediction['standard']
    try:
        obs = obs.rescale(pred.units)
        pre_error = np.abs((float(obs.magnitude)-float(pred.magnitude)))
        error = pre_error/np.abs(obs.magnitude)

    except:
        error = np.inf
    return error

def score_attr(dtcpop,pop):
    for i,d in enumerate(dtcpop):
        if not hasattr(pop[i],'dtc'):
            pop[i].dtc = None
        d.get_ss()
        pop[i].dtc = copy.copy(d)
    return dtcpop,pop
def get_dm(dtcpop,pop=None):
    if PARALLEL_CONFIDENT:
        NPART = min(npartitions,len(dtcpop))
        dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
        dtcpop = list(dtcbag.map(nuunit_dm_evaluation).compute())
    else:
        dtcpop = list(map(nuunit_dm_evaluation,dtcpop))
    if type(pop) is not type(None):
        dtcpop,pop = score_attr(dtcpop,pop)
    return dtcpop,pop



def bridge_passive(package):
    """
    Necessary because sciunit judge did not always work.
    """
    t,dtc = package

    model = new_model(dtc)
    t.setup_protocol(model)
    result = t.get_result(model)

    assert 'mean' in t.observation.keys()
    pred = None
    pred = t.extract_features(model,result)

    if type(pred) is type(None):
        return np.inf,dtc,pred

    if 'mean' in pred.keys():
        pred['standard'] = pred['mean']
    if 'value' in pred.keys():
        pred['standard'] = pred['value']
    if type(pred['standard']) is type(None):
        return np.inf,dtc,pred
    if type(pred['standard']) is not type(None):
        score = simple_error(t.observation,pred,t)
    return score, dtc, pred

from sciunit.scores.collections import ScoreArray

#from sciunit.scores.collections_m2m import  ScoreMatrixM2M,  ScoreArrayM2M#(pd.DataFrame, SciUnit, TestWeighted)

class OptMan():
    def __init__(self,tests, td=None, \
                backend = None,hc = None,\
                boundary_dict = None, \
                error_length=None,\
                protocol=None,\
                simulated_obs=None,\
                verbosity=None,\
                PARALLEL_CONFIDENT=None,\
                tsr=None):
        self.tests = tests
        if type(self.tests) is type(dict()):
            if 'name' in self.tests.keys():
                self.cell_name = tests['name']
                tests.pop('name',None)

        self.td = td
        if tests is not None:
            self.error_length = len(tests)
        self.backend = backend
        self.hc = hc
        self.boundary_dict= boundary_dict
        self.protocol = protocol
        self.julia = False
        self.simulated_data_tests = self.round_trip_test
        # note this is not effective at changing parallel behavior yet
        if PARALLEL_CONFIDENT not in globals():
            self.PARALLEL_CONFIDENT = False
        else:
            self.PARALLEL_CONFIDENT = PARALLEL_CONFIDENT
        if verbosity is None:
            self.verbose = 0
        else:
            self.verbose = verbosity
        if type(tsr) is not None:
            self.tsr = tsr

    def new_single_gene(self,dtc,td):
        random.seed(datetime.now())
        try:
            DO = SciUnitOptimisation(offspring_size = 1,
                error_criterion = [self.tests], boundary_dict = dtc.boundary_dict,
                                         backend = dtc.backend, selection = str('selNSGA'),simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)
        except:
            DO = SciUnitOptimisation(offspring_size = 1,
                error_criterion = [dtc.preds], boundary_dict = dtc.boundary_dict,
                                         backend = dtc.backend, selection = str('selNSGA'),simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)

        DO.setnparams(nparams = len(dtc.attrs), boundary_dict = self.boundary_dict)
        DO.setup_deap()
        gene = DO.set_pop(boot_new_random=1)

        dtc_ = self.update_dtc_pop(gene)#,self.td)
        #dtc_ = pop2dtc(gene,dtc_)
        return gene[0], dtc_[0]

    def optimize(self,free_params=None,NGEN=10,MU=10,seed_pop=None,hold_constant=None):
        from neuronunit.optimisation.optimisations import run_ga
        ranges = self.boundary_dict
        subset = OrderedDict()
        if type(free_params) is type(None):
            free_params = list(ranges.keys())

        if type(hold_constant) is type(None):
            temp = copy.copy(self.boundary_dict)
            for k in free_params:
                temp.pop(k,None)
            if len(temp)==0:
                hold_constant = None
            else:
                hold_constant = temp
            for k,v in hold_constant.items():
                hold_constant[k] = np.mean(v)


        ga_out,DO = run_ga(self.boundary_dict, NGEN, self.tests, free_params=free_params, \
                           backend=self.backend, MU = MU,  protocol=self.protocol,seed_pop = seed_pop,hc=hold_constant)
        self.DO = DO
        return ga_out
    def run_simple_grid(self,npoints=10,free_params=None):
        self.exhaustive = True
        from neuronunit.optimisation.exhaustive_search import sample_points, add_constant, chunks
        ranges = self.boundary_dict
        subset = OrderedDict()
        if type(free_params) is type(None):
            free_params = list(ranges.keys())
        for k,v in ranges.items():
            if k in free_params:
                subset[k] = ( np.min(ranges[k]),np.max(ranges[k]) )
        # The function of maps is to map floating point sample spaces onto a  monochromataic matrix indicies.
        subset = OrderedDict(subset)
        subset = sample_points(subset, npoints = npoints)
        grid_points = list(ParameterGrid(subset))
        if type(self.hc) is not type(None):
            grid_points = add_constant(self.hc,grid_points)#,self.td)
        self.td = list(grid_points[0].keys())

        if len(self.td) > 1:
            consumable = [ WSListIndividual(g.values()) for g in grid_points ]
        else:
            consumable = [ val for g in grid_points for val in g.values() ]
        grid_results = []
        self.td = list(self.td)
        if len(consumable) <= 16:
            results = self.update_deap_pop(consumable, self.tests, self.td,backend=self.backend)
            results_ = []
            for r in results:
                r.dtc = self.elephant_evaluation(r.dtc)
                results_.append(r)
            if type(results_) is not None:
                grid_results.extend(results_)

        if len(consumable) > 16:
            consumable = chunks(consumable,8)
            for sub_pop in consumable:
                #sub_pop = sub_pop
                results = self.update_deap_pop(sub_pop, self.tests, self.td)
                results_ = []
                for r in results:
                    r.dtc = self.elephant_evaluation(r.dtc)
                    results_.append(r)
                if type(results_) is not None:
                    grid_results.extend(results_)
        sorted_pop = sorted([ (gr.dtc.scores_ratio,gr.dtc.attrs,gr) for gr in grid_results ], key=lambda tup: tup[0])
        return grid_results, sorted_pop
    def get_allen(self,pop,dtcpop,tests,td,tsr=None):
        with open('waves.p','rb') as f:
            make_stim_waves = pickle.load(f)
        NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])
        for dtc in dtcpop: dtc.spike_number = tests['spike_count']['mean']
        for dtc in dtcpop: dtc.pre_obs = None
        for dtc in dtcpop: dtc.pre_obs = self.tests
        for dtc in dtcpop: dtc.tsr = tsr #not a property but an aim
        if PARALLEL_CONFIDENT:
            dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
            dtcpop = list(dtcbag.map(nuunit_allen_evaluation).compute())

        else:
            dtcpop = list(map(nuunit_allen_evaluation,dtcpop))

        pop = [pop[i] for i,d in enumerate(dtcpop) if type(d) is not type(None)]
        dtcpop = [d for d in dtcpop if type(d) is not type(None)]
        initial_length = len(pop)
        both = [(ind,dtc) for ind,dtc in zip(pop,dtcpop) if len(dtc.preds) > 6 ]
        both = [(ind,dtc) for ind,dtc in zip(pop,dtcpop) if dtc.ampl is not None ]

        if len(both):
            dtcpop = [i[1] for i in both]
            pop = [i[0] for i in both]
            delta = initial_length-len(both)
            for i in range(0,delta):
                pop.append(copy.copy(pop[0]))
                dtcpop.append(copy.copy(dtcpop[0]))
        else:
            for i,(ind,dtc) in enumerate(list(zip(pop,dtcpop))):
                target_current =  None
                dtc.error_length = 0
                while target_current is None:
                    ind,dtc = self.new_single_gene(dtc,self.td)
                    dtc.pre_obs = None
                    dtc.pre_obs = self.tests

                    observation_spike = {'range': [tests['spike_count']['mean']-3,tests['spike_count']['mean']+5] }
                    scs = SpikeCountRangeSearch(observation_spike)

                    model = new_model(dtc)
                    target_current = scs.generate_prediction(model)
                    dtc.ampl = None
                    dtc.error_length = len(dtc.preds)

                    if target_current is not None:
                        dtc.ampl = target_current['value']
                        dtc = prediction_current_and_features(dtc)

                        dtc = filter_predictions(dtc)

                        dtc.error_length = len(dtc.preds)

                pop[i] = ind
                dtcpop[i] = dtc

        if PARALLEL_CONFIDENT:
            dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
            dtcpop = list(dtcbag.map(nuunit_allen_evaluation).compute())

        else:
            dtcpop = list(map(nuunit_allen_evaluation,dtcpop))

        return pop, dtcpop



    def round_trip_test(self,
                        tests,
                        backend,
                        free_paramaters=None,
                        NGEN=None,
                        MU=None,
                        mini_tests=None,
                        stds=None):
        from neuronunit.optimisation.optimisations import run_ga


        '''
        # Inputs:
        #    -- tests, a list of NU test types,
        #    -- backend a string encoding what model, backend, simulator to use.
        # Outputs:
        #    -- a score, that should be close to zero larger is worse.
        # Synopsis:
        #    -- Given any models
        # check if the optimiser can find arbitarily sampeled points in
        # a parameter space, using only the information in the error gradient.
        # make some new tests based on internally generated data
        # as opposed to experimental data.
        '''

        out_tests = []

        if NGEN is None:
            NGEN = 10
        if MU is None:
            MU = 10
        ranges = MODEL_PARAMS[backend]
        #self.simulated_obs = True

        if self.protocol['allen']:
            dtc = False
            while dtc is False or type(new_tests) is type(None):
                _,rp,chosen_keys,random_param = process_rparam(backend)
                new_tests,dtc = self.make_simulated_observations(tests,backend,rp)
            test_origin_target=dtc
            observations = dtc.preds
            target_spikes = dtc.spike_number#+10
            observation_spike = {'value': target_spikes}
            new_tests = TSD(new_tests)

            new_tests.use_rheobase_score = False
            ga_out, DO = run_ga(ranges,NGEN,new_tests,free_params=rp.keys(), MU = MU, backend=backend, selection=str('selNSGA3'), protocol={'allen':True,'elephant':False,'tsr':[target_spikes-2,target_spikes+5]})
            ga_converged = [ p for p in ga_out['pf'] ]
            ga_out,ga_converged,test_origin_target,new_tests
            return ga_out,ga_converged,test_origin_target,new_tests,self.protocol['tsr']


        elif self.protocol['elephant']:
            new_tests = False
            while new_tests is False:
                dsolution,rp,chosen_keys,random_param = process_rparam(backend)
                (new_tests,dtc) = self.make_simulated_observations(tests,backend,rp,dsolution=dsolution)

                if type(new_tests) is not type(False):
                    if 'RheobaseTest' not in new_tests.keys():
                        continue
                    try:
                        dsolution.rheobase = new_tests['RheobaseTest'].observation
                    except:
                        dsolution.rheobase = new_tests['RheobaseTestP'].observation

            for k,v in new_tests.items():
                if type(v) is type({}):
                    v.observation['std'] = stds[k]
                    try:
                        v.observation['mean'] = v.observation['mean'].simplified
                    except:
                        v.observation['value'] = v.observation['value'].simplified

            # made not none through keyword argument.
            if type(mini_tests) is not type(None):
                results = {}
                mini_tests = {}

                for k,t in new_tests.items():
                    mini_tests[k] = t

                for k,v in mini_tests.items():
                    mt = {k: v}
                    if str('ReobaseTest') in new_tests.keys():
                        mt['RheobaseTest'] = new_tests['RheobaseTest']
                    if str('ReobaseTestP') in new_tests.keys():
                        mt['RheobaseTest'] = new_tests['RheobaseTestP']
                    try:
                        mt['RheobaseTest'] = new_tests['RheobaseTest']

                    except:
                        pass

                    temp = mt['RheobaseTest'].observation
                    if type(temp) is not type({'0':1}):
                        mt['RheobaseTest'].observation = {}
                        mt['RheobaseTest'].observation['value'] = temp#*pq.pA
                        mt['RheobaseTest'].observation['mean'] = temp#*pq.pA
                        mt['RheobaseTest'].observation['std'] = temp.magnitude*temp.units
                    if 'mean' not in temp.keys():
                        mt['RheobaseTest'].observation['mean'] = mt['RheobaseTest'].observation['value']
                    if 'std' not in temp.keys():
                        mt['RheobaseTest'].observation['std'] = mt['RheobaseTest'].observation['value']
                    new_tests = TSD(new_tests)
                    new_tests.use_rheobase_score = tests.use_rheobase_score

                    for v in mt.values():
                        if 'std' not in v.keys():
                            v['mean'] = v['std'] = v['value']


                    ga_out, DO = run_ga(ranges,NGEN,mt,free_params=rp.keys(), MU = MU, \
                        backend=backend,protocol={'elephant':True,'allen':False})
                    results[k] = copy.copy(ga_out['pf'][0].dtc.scores)

            else:
                if 'RheobaseTest' in new_tests.keys():
                    temp = new_tests['RheobaseTest'].observation
                else:
                    return

                if type(temp) is not type(dict()):
                    new_tests['RheobaseTest'].observation = {}
                    new_tests['RheobaseTest'].observation['value'] = temp#*pq.pA
                    new_tests['RheobaseTest'].observation['mean'] = temp#*pq.pA
                    new_tests['RheobaseTest'].observation['std'] = temp.magnitude*temp.units
                if 'mean' not in temp.keys():
                    new_tests['RheobaseTest'].observation['mean'] = new_tests['RheobaseTest'].observation['value']
                if 'std' not in temp.keys():
                    new_tests['RheobaseTest'].observation['std'] = new_tests['RheobaseTest'].observation['value']
                [(value.name,value.observation) for value in new_tests.values()]
                new_tests = TSD(new_tests)
                new_tests.use_rheobase_score = tests.use_rheobase_score

                for t in new_tests.values():
                    if 'value' in t.observation.keys():
                        t.observation['mean'] = t.observation['value']
                        t.observation['std'] = np.abs(t.observation['mean'])
                    if t.observation['std'] == 0.0:
                        t.observation['std'] = np.abs(t.observation['mean'])
                    try:
                        assert float(t.observation['std']) >0.0
                    except:
                        import pdb
                        pdb.set_trace()
                ga_out, DO = run_ga(ranges,
                                    NGEN,
                                    new_tests,
                                    free_params=chosen_keys,
                                    MU = MU,
                                    backend=backend,\
                    selection=str('selNSGA2'),protocol={'elephant':True,'allen':False})
                results = copy.copy(ga_out['pf'][0].dtc.scores)
            ga_converged = [ p.dtc for p in ga_out['pf'][0:2] ]
            test_origin_target = [ dsolution for i in range(0,len(ga_out['pf'])) ][0:2]
            if self.verbose:
                print(ga_out['pf'][0].dtc.scores)
                print(results)
                print(ga_out['pf'][0].dtc.attrs)

            #inject_and_plot(ga_converged,second_pop=test_origin_target,third_pop=[ga_converged[0]],figname='not_a_problem.png',snippets=True)
            return ga_out,ga_converged,test_origin_target,new_tests

    def grid_search(self,explore_ranges,test_frame,backend=None):
        '''
        Hopefuly this method can depreciate the whole file optimisation_management/exhaustive_search.py
        Well actually not quiete. This method does more than that. It iterates over multiple NeuroElectro datum entities.
        A more generalizable method would just act on one NeuroElectro datum entities.
        '''
        store_results = {}
        npoints = 8
        grid = ParameterGrid(explore_ranges)

        size = len(grid)
        temp = []
        if size > npoints:
            sparsify = np.linspace(0,len(grid)-1,npoints)
            for i in sparsify:
                temp.append(grid[int(i)])
            grid = temp
        dtcpop = []
        for local_attrs in grid:
            store_results[str(local_attrs.values())] = {}
            dtc = DataTC()
            #dtc.tests = use_test
            dtc.attrs = local_attrs

            dtc.backend = backend
            dtc.cell_name = backend

            dtcpop.append(dtc)


        for key, use_test in test_frame.items():
            for dtc in dtcpop:
                dtc.tests = use_test

            if 'RAW' in dtc.backend  or 'HH' in dtc.backend:#Backend:
                serial_faster = True # because of numba
                dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
                dtcpop = list(map(dtc_to_rheo,dtcpop))
            else:
                serial_faster = False
                dtcpop = list(map(dtc_to_rheo,dtcpop))
            dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
            dtcpop = list(dtcbag.map(self.format_test))
            dtcpop = [ dtc for dtc in dtcpop if type(dtc.rheobase) is not type(None) ]
            dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
            dtcpop = list(dtcbag.map(self.elephant_evaluation))
            for dtc in dtcpop:
                store_results[str(local_attrs.values())][key] = dtc.get_ss()
        df = pd.DataFrame(store_results)
        best_params = {}
        for index, row in df.iterrows():
            best_params[index] = row == row.min()
            best_params[index] = best_params[index].to_dict()
        seeds = {}
        for k,v in best_params.items():
            for nested_key,nested_val in v.items():
                if nested_val:
                    seed = nested_key
                    seeds[k] = seed
        with open(str(backend)+'_seeds.p','wb') as f:
            pickle.dump(seeds,f)
        return seeds, df

    @timer
    def pred_evaluation(self,dtc):
        # Inputs single data transport container modules, and neuroelectro observations that
        # inform test error error_criterion
        # Outputs Neuron Unit evaluation scores over error criterion
        dtc = copy.copy(dtc)
        # TODO
        # phase out model path:
        # via very reduced model
        if hasattr(dtc,'model_path'):
            dtc.model_path = path_params['model_path']
        #else:
        #    dtc.model_path = None
        #    dtc.model_path = path_params['model_path']
        dtc.preds = None
        dtc.preds = {}
        dtc = dtc_to_rheo(dtc)

        dtc = self.format_test(dtc)
        tests = dtc.tests
        if type(tests) is not type(dict()):
            tests = {t.name:t for t in tests}
        for k,t in tests.items():
            #if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
            t.params = dtc.protocols[k]
            test_and_models = (t, dtc)
            pred = pred_only(test_and_models)
            dtc.preds[str(t.name)] = pred
        return dtc

    @timer
    def score_specific_param_models(self,known_parameters,test_frame):
        '''
        -- Inputs: tests, simulator backend, the parameters to explore
        -- Outputs: data frame of index paramaterized model, column tests suite aggregate average.
        '''
        try:
            from dask import dataframe as pdd
            df = pdd.DataFrame(index=list(test_frame.keys()),columns=list(known_parameters.keys()))
        except:
            df = pd.DataFrame(index=list(test_frame.keys()),columns=list(known_parameters.keys()))

        backend = self.backend
        for l,(key, use_test) in enumerate(self.tests.items()):
            if RheobaseTest in use_test.keys():
                use_test.use_rheobase_score = True
            else:
                use_test.use_rheobase_score = False
            OM = OptMan(use_test,backend=backend,\
                        boundary_dict=MODEL_PARAMS[backend],\
                        protocol={'elephant':True,'allen':False,'dm':False},)#'tsr':spk_range})
            dtcpop = []
            for k,v in known_parameters.items():
                temp = {}
                temp[str(v)] = {}
                dtc = DataTC()
                dtc.attrs = v
                dtc.backend = backend
                dtc.cell_name = 'vanilla'
                dtc.tests = copy.copy(use_test)
                dtc = dtc_to_rheo(dtc)
                dtc.tests = dtc.format_test()
                dtcpop.append(dtc)
            try:
                bagged = db.from_sequence(dtcpop,npartitions=npartitions)
                dtcpop = list(bagged.map(OM.elephant_evaluation))

            except:
                dtcpop = list(map(OM.elephant_evaluation,dtcpop))
            for i,j in enumerate(dtcpop):
                df.iloc[l][i] = np.sum(list(j.scores.values()))/len(list(j.scores.values()))

        return df
    #@timer
    def pred_std(self,pred,t):
        take_anything = list(pred.values())[0]
        if take_anything is None or type(take_anything) is type(int()):
            take_anything = list(pred.values())[1]

        pred['std'] = t.observation['std']# 15*take_anything.magnitude * take_anything.units
        return pred


    def preprocess(self,dtc):
        tests = dtc.tests

        if isinstance(dtc.rheobase,type(None)) or type(dtc.rheobase) is type(None):
            dtc = allocate_worst(tests, dtc)
        else:
            for t in tests:
                k = str(t.name)
                # it's critical that paramaters are assigned here
                t.params = dtc.protocols[k]

                if "RheobaseTest" in  t.name:
                    t.score_type = scores.ZScore
                    # The code block below
                    # is prefeable but it won't lead to scoring
                    """
                    if str("BHH") in dtc.backend or str("ADEXP") in dtc.backend or str("GLIF") in dtc.backend:
                        t = RheobaseTestP(t.observation)
                        # it's critical that paramaters are assigned here

                        t.params = dtc.protocols['RheobaseTest']
                        t.passive = False
                    """
                try:
                    assert hasattr(self.tests,'use_rheobase_score')
                except:
                    print('warning please add whether or not model should be scored on rheobase to protocol')
                    self.tests.use_rheobase_score = True
                if self.tests.use_rheobase_score == False and "RheobaseTest" in str(k):
                    continue
                t.params = dtc.protocols[k]
                if not 'std' in t.observation.keys():
                    t.observation['std'] = t.observation['mean']
                if float(t.observation['std']) == 0.0:
                    t.observation['std'] = t.observation['mean']
        return tests
    def elephant_evaluation(self,dtc):
        # Inputs single data transport container modules, and neuroelectro observations that
        # inform test error error_criterion
        # Outputs Neuron Unit evaluation scores over error criterion
        model = dtc.dtc_to_model()
        if not hasattr(dtc,'scores') or dtc.scores is None:
            dtc.scores = None
            dtc.scores = {}
        dtc.tests = self.preprocess(dtc)
        if PARALLEL_CONFIDENT is False:
            suite = TestSuite(dtc.tests)
            #try:
            dtc.SM = suite.judge(model,parallel=False,log_norm=True)
        else:
            scores_ = []
            for t in dtc.tests:
                score = t.judge(model)
                if score.get_raw() == 0:
                     t.score_type = scores.ZScore
                     score = t.judge(model)
                     score.log_norm_score
                scores_.append(score)
            dtc.SM = ScoreArray(tests, scores_)
        dtc.obs_preds = pd.DataFrame([{t.name:t.observation for t in dtc.tests},{t.name:t.prediction for t in dtc.tests}])
        assert dtc.SM is not None
        return dtc


        @timer
        def serial_route(self,pop,td,tests):
            '''
            parallel list mapping only works with an iterable collection.
            Serial route is intended for single items.
            '''
            if type(dtc.rheobase) is type(None):
                for t in tests:
                    dtc.scores[t.names] = 1.0
                    dtc.get_ss()
            else:
                dtc = self.format_test((dtc,tests))
                dtc = self.elephant_evaluation((dtc,tests))

            return pop, dtc
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @timer
    def format_test(self,dtc):
        # pre format the current injection dictionary based on pre computed
        # rheobase values of current injection.
        # This is much like the hooked method from the old get neab file.
        if type(dtc) is type(str()):
            print('error dtc is string')
            import pdb
            pdb.set_trace()
        dtc.protocols = {}
        if not hasattr(dtc,'tests'):
            dtc.tests = copy.copy(self.tests)

        if type(dtc.tests) is type(dict()):
            for t in dtc.tests.values():
                assert 'std' in t.observation.keys()


        if hasattr(dtc.tests,'keys'):# is type(dict):
            tests = [key for key in dtc.tests.values()]
            dtc.tests = switch_logic(tests)#,self.tests.use_rheobase_score)
        else:
            dtc.tests = switch_logic(dtc.tests)

        for v in dtc.tests:
            k = v.name
            dtc.protocols[k] = {}
            if hasattr(v,'active'): #['protocol']:
                if v.passive == False and v.active == True:
                    keyed = dtc.protocols[k]#.params
                    dtc.protocols[k] = active_values(keyed,dtc.rheobase)
            if hasattr(v,'passive'):
                if v.passive == True and v.active == False:
                    keyed = dtc.protocols[k]#.params
                    dtc.protocols[k] = passive_values(keyed)
            if v.name in str('RestingPotentialTest'):

                dtc.protocols[k]['injected_square_current']['amplitude'] = 0.0*pq.pA
        #import pdb
        #pdb.set_trace()

        return dtc
    @timer
    def make_simulated_observations(self,original_test_dic,backend,random_param,dsolution=None):
        #self.simulated_obs = True
        # to be used in conjunction with round_trip_test below.

        if dsolution is None:
            dtc = DataTC()
            dtc.attrs = random_param
            dtc.backend = copy.copy(backend)
        else:
            dtc = dsolution
        if self.protocol['elephant']:

            if str('RheobaseTest') in original_test_dic.keys():
                dtc = get_rh(dtc,original_test_dic['RheobaseTest'])
                if type(dtc.rheobase) is type(dict()):
                    if dtc.rheobase['value'] is None:
                        return False, dtc
                elif type(dtc.rheobase) is type(float(0.0)):
                    pass

            dtc = make_new_random(dtc, copy.copy(backend))
            xtests = list(copy.copy(original_test_dic).values())

            dtc.tests = xtests
            simulated_observations = {}
            xtests = [t for t in xtests if 'mean' in t.observation.keys() or 'value' in t.observation.keys() ]
            for i,t in enumerate(xtests):
                if 'mean' in t.observation.keys():
                    simulated_observations[t.name] = copy.copy(t.observation['mean'])
                elif 'value' in t.observation.keys():
                    simulated_observations[t.name] = copy.copy(t.observation['value'])
                else:
                    return (dtc,False)

            #if False not in set(means):
            #    simulated_observations = {t.name:copy.copy(t.observation['mean']) for t in xtests}
            #    simulated_observations = {k:v for k,v in simulated_observations.items() if v is not None}
            dtc.observation = simulated_observations
            dtc = self.pred_evaluation(dtc)

            simulated_observations = {k:p for k,p in dtc.preds.items() if type(k) is not type(None) and type(p) is not type(None) }

            while len(dtc.preds)!= len(simulated_observations):
                dtc = make_new_random(dtc, copy.copy(backend))
                dtc = self.pred_evaluation(dtc)
                dtc.tests = xtests
                simulated_observations = {k:p for k,p in dtc.preds.items() if type(k) is not type(None) and type(p) is not type(None) }


                if str("RheobaseTest") in simulated_observations.keys():
                    temp = copy.copy(simulated_observations['RheobaseTest'])
                    simulated_observations['RheobaseTest'] = {}

                    if type(temp) is type(dict()):
                        if 'value' in temp.keys():
                            simulated_observations['RheobaseTest']['value'] = temp['value']
                    else:
                        simulated_observations['RheobaseTest']['value'] = temp
                    break
                else:
                    continue

            simulated_observations = {k:p for k,p in simulated_observations.items() if type(k) is not type(None) and type(p) is not type(None) }
            if self.verbose:
                print(simulated_observations)
            simulated_tests = {}
            for k in xtests:
                if k.name in simulated_observations.keys():
                    k.observation = simulated_observations[k.name]
                    simulated_tests[k.name] = k

            if self.verbose:
                print('try generating rheobase from this test')
            return simulated_tests, dtc


        if self.protocol['allen']:
            #target_current = None
            dtc = DataTC()
            dtc.backend = backend
            dtc.pre_obs = xtests
            dtc.attrs = random_p(dtc.backend)
            dtc.preds = {}
            target_current = None
            while target_current is None or not hasattr(dtc,'spike_cnt'):# is None:
                dtc.attrs = random_p(dtc.backend)
                try:
                    with open('waves.p','wb') as f:
                        make_stim_waves = pickle.load(f)
                except:
                    make_stim_waves = make_stim_waves_func()

                from neuronunit.tests.target_spike_current import SpikeCountSearch, SpikeCountRangeSearch

                observation_range = {'range': [8, 15]}

                #if dtc.backend is str("GLIF"):
                try:
                    observation_range={}
                    observation_range['value'] = 15.0
                    scs = SpikeCountSearch(observation_range)
                    model = new_model(dtc)
                    assert model is not None
                    target_current = scs.generate_prediction(model)
                except:
                    observation_range = {'range': [8, 15]}

                    scs = SpikeCountRangeSearch(observation_range)
                    model = new_model(dtc)

                    assert model is not None
                    target_current = scs.generate_prediction(model)


                dtc.ampl = None
                if target_current is not None:

                    dtc.ampl = target_current['value']
                else:
                    dtc.rheobase = {}
                    dtc.rheobase = None
                    while type(dtc.rheobase) is type(None):
                        dtc = dtc_to_rheo(dtc)
                        if type(dtc.rheobase) is type(None):
                            continue
                    try:
                        dtc.ampl = dtc.rheobase['value'] * 1.5
                    except:
                        dtc.ampl = dtc.rheobase * 1.5


                target_current = dtc.ampl
                dtc.pre_obs = None
                returned = just_allen_predictions(dtc)
                try:
                    dtc,_,_ = returned
                except:
                    dtc = False
                    return None,dtc


                dtc = filter_predictions(dtc)
            dtc.spike_number = dtc.spike_cnt
            target_spikes = dtc.spike_cnt
            dtc.preds['spike_count'] ={}
            dtc.preds['spike_count']['mean'] = target_spikes
            dtc.preds['current'] = {}
            dtc.preds['current']['mean'] = dtc.ampl
            dtc.pre_obs = dtc.preds
            return dtc.preds, dtc
    #@timer
    def update_dtc_pop(self,pop):
        '''
        inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
        outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
        Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
        compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
        If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
        corresponding virtual model objects.
        '''
        if self.backend is not None:
            _backend = self.backend
        if isinstance(pop, Iterable):# and type(pop[0]) is not type(str('')):
            xargs = zip(pop,repeat(self.td),repeat(self.backend))
            npart = np.min([multiprocessing.cpu_count(),len(pop)])
            bag = db.from_sequence(xargs, npartitions = npart)
            dtcpop = list(bag.map(transform).compute())
            if self.verbose:
                print(dtcpop)
            assert len(dtcpop) == len(pop)
            for dtc in dtcpop:
                dtc.backend = self.backend

            if self.hc is not None:
                for d in dtcpop:
                    d.attrs.update(self.hc)

                #dtc.boundary_dict = None
                #dtc.boundary_dict = self.boundary_dict
            return dtcpop
        else:
            ind = pop
            for i in ind:
                #i.td = td
                i.backend = str(_backend)
            # above replaces need for this line:
            xargs = (ind,td,repeat(backend))
            # In this case pop is not really a population but an individual
            # but parsimony of naming variables
            # suggests not to change the variable name to reflect this.
            dtc = [ transform(xargs) ]
            dtc.boundary_dict = None
            dtc.boundary_dict = self.boundary_dict
            return dtc

    @timer
    def init_pop(self,pop, tests):

        dtcpop = list(self.update_dtc_pop(pop))
        for d in dtcpop:
            d.tests = tests
            if self.backend is not None:
                d.backend = self.backend

        if not hasattr(self,'exhaustive'):
            if self.hc is not None:
                for d in dtcpop:
                    if self.verbose:
                        print(self.hc)
                    d.attrs.update(self.hc)

        return pop, dtcpop
    def defend_against_zero_std(self,dtcpop):
        for dtc in dtcpop:
            if type(dtc.tests) is type(dict()):
                for t in dtc.tests.values():
                    assert 'std' in t.observation.keys()
    @timer
    def obtain_rheobase(self,pop,tests):#, td, tests):
        '''
        Calculate rheobase for a given population pop
        Ordered parameter dictionary td
        and rheobase test rt
        '''


        _, dtcpop = self.init_pop(pop, tests)
        for d in dtcpop:
            d.tests = tests

        for dtc in dtcpop:

            if type(dtc.tests) is type(dict()):
                for t in dtc.tests.values():
                    assert 'std' in t.observation.keys()

            elif type(dtc.tests) is type(list()):
                for t in dtc.tests:
                    assert 'std' in t.observation.keys()

        if 'RAW' in self.backend  or 'HH' in self.backend or str('ADEXP') in self.backend:
            self.defend_against_zero_std(copy.copy(dtcpop))
            dtcpop = list(map(dtc_to_rheo,dtcpop))
            for dtc in dtcpop:
                if type(dtc.tests) is type(dict()):
                    for t in dtc.tests.values():
                        assert 'std' in t.observation.keys()

            dtcpop = list(map(self.format_test,dtcpop))
            dtcpop = [d for d in dtcpop if d is not None]
        else:
            dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
            dtcpop = list(dtcbag.map(dtc_to_rheo))


        if not hasattr(pop[0],'rheobase'):
            self.julia = True
            pop_ = []
            for ind,d in zip(pop,dtcpop):
                if type(ind) is not type(WSListIndividual()):
                    temp = ind.tolist()
                    temp_ind = WSListIndividual(obj_size=len(self.tests))
                    temp_ind.extend(temp)
                    ind = temp_ind
                    ind.dtc = d
                    pop_.append(ind)
            pop = pop_
        for ind,d in zip(pop,dtcpop):
            if type(d.rheobase) is not type(None):
                ind.rheobase = d.rheobase
                d.rheobase = d.rheobase
            else:
                ind.rheobase = None
                d.rheobase = None
        return pop, dtcpop
    def serial_dtc(self,dtcpop,b=False):
        parallel_dtc = []
        for dtc in dtcpop:
            parallel_dtc.append(DataTC());

            parallel_dtc[-1].attrs = dtc.attrs;
            parallel_dtc[-1].backend = dtc.backend;
            parallel_dtc[-1].rheobase = dtc.rheobase;
            parallel_dtc[-1].protocols = dtc.protocols
            # suggesting that passive tests not always picklable
            # as the passive tests associated with brian backend objects have
            # persistant weak references that cannot be used.
            parallel_dtc[-1].tests = []

            for t in dtc.tests:
                if t.passive==b:
                    parallel_dtc[-1].tests.append(t)
        return parallel_dtc
    @timer
    def parallel_route(self,pop,dtcpop,tests,td):
        NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])

        if self.protocol['allen']:
            pop, dtcpop = self.get_allen(pop,dtcpop,tests,td,tsr=self.protocol['tsr'])
            pop = [pop[i] for i,d in enumerate(dtcpop) if type(d) is not type(None)]
            dtcpop = [d for d in dtcpop if type(d) is not type(None)]
            return pop, dtcpop

        elif str('dm') in self.protocol.keys():
            if self.protocol['dm']:
                pop, dtcpop = get_dm(pop,dtcpop)
                return pop, dtcpop

        elif self.protocol['elephant']:
            for d in dtcpop:
                d.tests = copy.copy(self.tests)


            if PARALLEL_CONFIDENT:# and self.backend is not str('ADEXP'):
                passed = False
                try:
                    dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
                    dtcpop = list(dtcbag.map(self.format_test).compute())
                    passed = True
                except:
                    dtcpop = list(map(self.format_test,dtcpop))

                dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
                dtcpop = list(dtcbag.map(self.elephant_evaluation).compute())

                for d in dtcpop:
                    assert hasattr(d, 'tests')
                    assert dtc.SM is not None

                for d in dtcpop:
                    d.tests = copy.copy(self.tests)

            if not PARALLEL_CONFIDENT:
                dtcpop = list(map(self.format_test,dtcpop))
                #if self.backend == 'ADEXP':
                dtcpop = list(map(self.elephant_evaluation,dtcpop))

                for d in dtcpop:
                    d.tests = copy.copy(self.tests)
                for d in dtcpop:
                    assert hasattr(d, 'tests')
                    assert d.SM is not None


            for d in dtcpop:
               if not hasattr(d, 'tests'):
                 print(Error('broken no test in dtc'))
        return pop, dtcpop
    def test_runner(self,pop,td,tests):
        if self.protocol['elephant']:
            if type(tests) is type(dict()):
                for t in tests.values():
                    assert 'std' in t.observation.keys()

            elif type(tests) is type(list()):
                for t in tests:
                    assert 'std' in t.observation.keys()

            pop_, dtcpop = self.obtain_rheobase(pop, tests)

            for ind,dtc in zip(pop,dtcpop):
                dtc.error_length = self.error_length


            if not hasattr(self,'exhaustive'):
                # there are many models, which have no actual rheobase current injection value.
                # filter, filters out such models,
                # gew genes, add genes to make up for missing values.
                # delta is the number of genes to replace.

                pop, dtcpop = self.make_up_lost(copy.copy(pop_), dtcpop, td)
            else:
                pop,dtcpop = self.parallel_route(pop, dtcpop, tests, td)#, clustered=False)
                both = [(ind,dtc) for ind,dtc in zip(pop,dtcpop) if dtc.scores is not None]
                for ind,d in both:
                    if not hasattr(ind,'fitness'):
                        ind.fitness = []#copy.copy(pop_[0].fitness)

        elif self.protocol['allen']:

            pop, dtcpop = self.init_pop(pop, tests)
            for ind,dtc in zip(pop,dtcpop):
                dtc.error_length = self.error_length

            for ind,d in zip(pop,dtcpop):
                d.error_length = self.error_length
                ind.error_length = self.error_length
        pop,dtcpop = self.parallel_route(pop, dtcpop, tests, td)#, clustered=False)
        both = [(ind,dtc) for ind,dtc in zip(pop,dtcpop) if dtc.SM.score is not None]
        for ind,d in both:
            ind.dtc = None
            ind.dtc = d
            if d.SM.score is not None:
                ind = copy.copy(both[0][0])
                d = copy.copy(both[0][1])

            if not hasattr(ind,'fitness'):
                ind.fitness = copy.copy(pop_[0].fitness)
                for i,v in enumerate(list(ind.fitness.values)):
                    ind.fitness.values[i] = list(ind.dtc.SM.score.values())[i]
        #import pdb
        #pdb.set_trace()
        pop = [ ind for ind,d in zip(pop,dtcpop) if d.SM.score is not None ]
        dtcpop = [ d for ind,d in zip(pop,dtcpop) if d.SM.score is not None ]
        return pop,dtcpop

    @timer
    def make_up_lost(self,pop,dtcpop,td):
        '''
        make new genes, actively replace genes for unstable solutions.
        Alternative: let gene pool shrink momentarily, risks in breading.
        '''
        before = len(pop)
        fitness_attr = pop[0].fitness
        spare = copy.copy(dtcpop)

        (pop,dtcpop) = filtered(pop,dtcpop)
        after = len(pop)
        delta = before-after
        if not delta:
            return pop, dtcpop
        if delta:

            cnt = 0
            while delta:
                pop_,dtcpop_ = self.boot_new_genes(delta,spare,td)
                for dtc,ind in zip(pop_,dtcpop_):
                    ind.from_imputation = None
                    ind.from_imputation = True
                    dtc.from_imputation = True
                    dtc.tests = copy.copy(self.tests)
                    dtc = self.format_test(dtc)
                    ind.dtc = dtc

                pop_ = [ p for p in pop_ if len(p)>1 ]

                pop.extend(pop_)
                dtcpop.extend(dtcpop_)
                (pop,dtcpop) = filtered(pop,dtcpop)
                for i,p in enumerate(pop):
                    if not hasattr(p,'fitness'):
                        p.fitness = fitness_attr

                after = len(pop)
                delta = before-after
                if delta:
                    continue
                else:
                    break

            return pop, dtcpop
    @timer
    def update_deap_pop(self,pop, tests, td, backend = None,hc = None,boundary_dict = None, error_length=None):
        '''
        Returned neuronunit scored DataTransportContainers (dtcpop).
        This method converts a population of genes to a population of Data Transport Containers,
        Which act as communicatable data types for storing model attributes.
        Rheobase values are found on the DTCs
        DTCs for which a rheobase value of x (pA)<=0 are filtered out
        DTCs are then scored by neuronunit, using neuronunit models that act in place.
        '''
        if len(pop)==0:
            raise Exception('User error population size set to 0')
        if td is not None:
            self.td = td

        if hc is not None:
            self.hc = hc

        if backend is not None:
            self.backend = None
            self.backend = backend
        if boundary_dict is not None:
            self.boundary_dict = None
            self.boundary_dict = boundary_dict
        for _ in range(0,len(pop)):
            if error_length is not None:
                self.error_length = None
                self.error_length = error_length
        pop, dtcpop = self.test_runner(pop,td,self.tests)

        for p,d in zip(pop,dtcpop):
            p.dtc = d
            p.error_length = self.error_length
            p.backend = self.backend
        if self.julia:
            fitnesses = [ list(dtc.errors.values()) for dtc in dtcpop ]
            return fitnesses, pop, dtcpop

        return pop
    @timer
    def boot_new_genes(self,number_genes,dtcpop,td):
        '''
        Boot strap new genes to make up for completely called onesself.
        '''

        random.seed(datetime.now())
        DO = SciUnitOptimisation(MU = number_genes,
                                 error_criterion = self.tests, boundary_dict =self.boundary_dict,
                                 backend = self.backend, selection = str('selNSGA'),protocol = self.protocol)#,, boundary_dict = ss, elite_size = 2, hc=hc)

        DO.setnparams(nparams = len(dtcpop[0].attrs), boundary_dict = self.boundary_dict)
        DO.setup_deap()
        if 1==number_genes:
            pop = DO.set_pop(boot_new_random=5)
        if 1<number_genes and number_genes<5:
            pop = DO.set_pop(boot_new_random=5)

        else:
            pop = DO.set_pop(boot_new_random=number_genes)
        dtcpop_ = self.update_dtc_pop(pop)
        dtcpop_ = list(map(dtc_to_rheo,dtcpop_))
        for i,ind in enumerate(pop):
            pop[i].rheobase = dtcpop_[i].rheobase
            pop[i].dtc = dtcpop_[i]

        pop = pop[0:number_genes]
        dtcpop_ = dtcpop_[0:number_genes]

        return pop, dtcpop_


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
