
# Its not that this file is responsible for doing plotting,
# but it calls many modules that are, such that it needs to pre-empt
import warnings
import dask
from tqdm import tqdm
SILENT = True
if SILENT:
    warnings.filterwarnings("ignore")
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

import cython
import matplotlib.pyplot as plt
import numpy as np
import dask.bag as db
import dask.delayed as delay
import pandas as pd
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
import math
import quantities as pq
import numpy

from deap import creator
from deap import base
import array
import copy
from frozendict import frozendict

from neuronunit.optimization.data_transport_container import DataTC
from itertools import repeat
from neuronunit.tests.base import AMPL, DELAY, DURATION
from collections.abc import Iterable
from neuronunit.tests.target_spike_current import SpikeCountSearch, SpikeCountRangeSearch
import neuronunit.capabilities.spike_functions as sf
from sciunit import TestSuite


import sciunit
import deap
import time
import seaborn as sns
import quantities as pq
import random
import pandas as pd
from neuronunit.tests.target_spike_current import SpikeCountSearch, SpikeCountRangeSearch
from neuronunit.tests.fi import RheobaseTest
from sciunit import scores
import seaborn as sns
sns.set(context="paper", font="monospace")
from jithub.models import model_classes

import quantities as qt
import bluepyopt.ephys as ephys
from bluepyopt.parameters import Parameter
import quantities as pq
PASSIVE_DURATION = 500.0*pq.ms
PASSIVE_DELAY = 200.0*pq.ms

try:
    import plotly.offline as py
except:
    warnings.warn('plotly')
try:
    import plotly
    plotly.io.orca.config.executable = '/usr/bin/orca'
except:
    print('silently fail on plotly')


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

def set_up_obj_tests(model_type):
    backend = model_type
    with open('processed_multicellular_constraints.p','rb') as f: test_frame = pickle.load(f)
    stds = {}
    for k,v in TSD(test_frame['Neocortex pyramidal cell layer 5-6']).items():
        temp = TSD(test_frame['Neocortex pyramidal cell layer 5-6'])[k]
        stds[k] = temp.observation['std']
    cloned_tests = copy.copy(test_frame['Neocortex pyramidal cell layer 5-6'])
    cloned_tests = TSD(cloned_tests)
    #{'RestingPotentialTest':cloned_tests['RestingPotentialTest']}
    OM = jrt(cloned_tests,backend,protocol={'elephant':True,'allen':False})
    return OM
def test_all_objective_test(free_parameters,
                            model_type="IZHI",
                            protocol=None,tract=True):
    ''' a wrapper and a short cut to
    jump to a simulated data context
    '''
    results = {}
    tests = {}
    OM = set_up_obj_tests(model_type)
    #if test_keys is None:

    if tract == False:
        test_keys = ["RheobaseTest","TimeConstantTest","RestingPotentialTest",
        "InputResistanceTest","CapacitanceTest","InjectedCurrentAPWidthTest",
        "InjectedCurrentAPAmplitudeTest","InjectedCurrentAPThresholdTest"]
    else:
        test_keys = ["RheobaseTest","TimeConstantTest","RestingPotentialTest",
        "InputResistanceTest","CapacitanceTest"]#,"FITest"]


    simulated_data_tests, OM, target = OM.make_sim_data_tests(
        model_type,
        free_parameters=free_parameters,
        test_key=test_keys,
        protocol=protocol
        )
    stds = {}


    for k,v in simulated_data_tests.items():

        keyed = which_key(simulated_data_tests[k].observation)
        if k == str('RheobaseTest'):
            mean = simulated_data_tests[k].observation[keyed]
            std = simulated_data_tests[k].observation['std']
            x = np.abs(std/mean)
        if k == str('TimeConstantTest') or k == str('CapacitanceTest') or k == str('InjectedCurrentAPWidthTest'):
            # or k == str('InjectedCurrentAPWidthTest'):
            mean = simulated_data_tests[k].observation[keyed]
            simulated_data_tests[k].observation['std'] = np.abs(mean)*2.0
        elif k == str('InjectedCurrentAPThresholdTest') or k == str('InjectedCurrentAPAmplitudeTest'):
            mean = simulated_data_tests[k].observation[keyed]
            simulated_data_tests[k].observation['std'] = np.abs(mean)*2.0
        stds[k] = (x,mean,std)
    target.tests = simulated_data_tests
    model = target.dtc_to_model()
    for t in simulated_data_tests.values():
        try:
            score0 = t.judge(target.dtc_to_model())
        except:
            pass
            #pdb.set_trace()
        score1 = target.tests[t.name].judge(target.dtc_to_model())
        assert float(score0.score)==0.0
        assert float(score1.score)==0.0
    tests = TSD(copy.copy(simulated_data_tests))
    check_tests = copy.copy(tests)
    return tests, OM, target



def make_ga_DO(explore_edges, max_ngen, test, \
        free_parameters = None, hc = None,
        MU = None, seed_pop = None, \
           backend = str('IZHI'),protocol={'allen':False,'elephant':True}):

    # construct an DEAP Optimization Object, suitable for this test class and caching etc.

    ss = {}
    if type(free_parameters) is type(dict()):
        if 'dt' in free_parameters:
            free_parameters.pop('dt')
        if 'Iext' in free_parameters:
            free_parameters.pop('Iext')
    else:
        free_parameters = [f for f in free_parameters if str(f) != 'Iext' and str(f) != str('dt')]
    for k in free_parameters:
        if not k in explore_edges.keys() and k != str('Iext') and k != str('dt'):

            ss[k] = explore_edges[k]
        else:
            ss[k] = explore_edges[k]
    if type(MU) == type(None):
        MU = 2**len(list(free_parameters))
    else:
        MU = MU
    max_ngen = int(np.floor(max_ngen))
    if not isinstance(test, Iterable):
        test = [test]
    from neuronunit.optimization.optimizations import SciUnitoptimization
    from bluepyopt.optimizations import DEAPoptimization
    DO = SciUnitoptimization(MU = MU, tests = test,\
         boundary_dict = ss, backend = backend, hc = hc, \
                             protocol=protocol)

    if seed_pop is not None:
        # This is a re-run condition.
        DO.setnparams(nparams = len(free_parameters), boundary_dict = ss)

        DO.seed_pop = seed_pop
        DO.setup_deap()
        DO.error_length = len(test)

    return DO


class TSD(dict):
    """
    Test Suite Dictionary class

    A container for sciunit tests, Indexable by dictionary keys.

    Contains a method called optimize.
    """
    def __init__(self,tests={},use_rheobase_score=True):
       self.DO = None
       self.use_rheobase_score = use_rheobase_score
       #self.elaborate_plots  = elaborate_plots
       self.backend = None
       self.three_step = None
       if type(tests) is TestSuite:
           tests = OrderedDict({t.name:t for t in tests.tests})
       if type(tests) is type(dict()):
           pass
       if type(tests) is type(list()):
          tests = OrderedDict({t.name:t for t in tests})
       super(TSD,self).__init__()
       self.update(tests)
       if 'allen_hack' in self.keys():
           self.three_step = allen_hack
           self.pop('allen_hack',None)



       if 'name' in self.keys():
           self.cell_name = tests['name']
           self.pop('name',None)
       else:
           self.cell_name = 'simulated data'

    def to_pickleable_dict(self):

        # A pickleable version of instance object.
        # https://joblib.readthedocs.io/en/latest/


        # This might work joblib.dump(self, filename + '.compressed', compress=True)
        # somewhere in job lib there are tools for pickling more complex objects
        # including simulation results.
        del self.ga_out.DO
        del self.DO
        return {k:v for k,v in self.items() }

    def optimize(self,**kwargs):
        import shelve
        defaults = {'param_edges':None,
                    'backend':None,\
                    'protocol':{'allen': False, 'elephant': True},\
                    'MU':5,\
                    'NGEN':5,\
                    'free_parameters':None,\
                    'seed_pop':None,\
                    'hold_constant':None,
                    'plot':False,'figname':None,
                    'use_rheobase_score':self.use_rheobase_score,
                    'ignore_cached':False
                    }
        defaults.update(kwargs)
        kwargs = defaults
        d = shelve.open('opt_models_cache')  # open -- file may get suffix added by low-level
        query_key = str(kwargs['NGEN']) +\
        str(kwargs['free_parameters']) +\
        str(kwargs['backend']) +\
        str(kwargs['MU']) +\
        str(kwargs['protocol']) +\
        str(kwargs['hold_constant'])
        flag = query_key in d

        if flag and not kwargs['ignore_cached']:
            ###
            # Hack
            ###
            #creator.create("FitnessMin", base.Fitness, weights=tuple(-1.0 for i in range(0,10)))
            #creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
            ###
            # End hack
            ###
            ga_out = d[query_key]

            d.close()
            del d
            return ga_out
        else:
            d.close()
            del d
            if type(kwargs['param_edges']) is type(None):
                from neuronunit.optimization import model_parameters
                param_edges = model_parameters.MODEL_PARAMS[kwargs['backend']]
            if type(kwargs['free_parameters']) is type(None):
                if type(kwargs['param_edges']) is not type(None):
                    free_parameters=kwargs['param_edges'].keys()
                else:
                    from neuronunit.optimization import model_parameters
                    free_parameters = model_parameters.MODEL_PARAMS[kwargs['backend']].keys()
            else:
                free_parameters = kwargs['free_parameters']

            if kwargs['hold_constant'] is None:
                if len(free_parameters) < len(param_edges):
                    pass
            self.DO = make_ga_DO(param_edges, \
                                kwargs['NGEN'], \
                                self, \
                                free_parameters=free_parameters, \
                                backend=kwargs['backend'], \
                                MU = kwargs['MU'], \
                                protocol=kwargs['protocol'],
                                seed_pop = kwargs['seed_pop'], \
                                hc=kwargs['hold_constant']
                                )
            self.MU = self.DO.MU = kwargs['MU']
            self.NGEN = self.DO.NGEN = kwargs['NGEN']

            ga_out = self.DO.run(NGEN = self.DO.NGEN)
            self.backend = kwargs['backend']
            return ga_out

    def display(self):
        from IPython.display import display
        if hasattr(self,'ga_out'):
            return display(self.ga_out['pf'][0].dtc.obs_preds)
        else:
            return None


@cython.boundscheck(False)
@cython.wraparound(False)
def random_p(backend):
    ranges = MODEL_PARAMS[backend]
    import numpy, time
    date_int = int(time.time())
    numpy.random.seed(date_int)
    random_param1 = {} # randomly sample a point in the viable parameter space.
    for k in ranges.keys():
        sample = random.uniform(ranges[k][0], ranges[k][1])
        random_param1[k] = sample
    return random_param1

@cython.boundscheck(False)
@cython.wraparound(False)
def process_rparam(backend,free_parameters):
    random_param = random_p(backend)
    if 'GLIF' in str(backend):
        random_param['init_AScurrents'] = [0.0,0.0]
        random_param['asc_tau_array'] = [0.3333333333333333,0.01]
        rp = random_param
    else:
        random_param.pop('Iext',None)
        rp = random_param
    if free_parameters is not None:
        reduced_parameter_set = {}
        for k in free_parameters:
            reduced_parameter_set[k] = rp[k]
        rp = reduced_parameter_set
    dsolution = DataTC(backend =backend,attrs=rp)
    temp_model = dsolution.dtc_to_model()
    dsolution.attrs = temp_model.default_attrs
    dsolution.attrs.update(rp)
    return dsolution,rp,None,random_param
'''
def check_test(new_tests):
    replace = False
    for k,t in new_tests.items():
        if type(t.observation['value']) is type(None):
            replace = True
            return replace
'''
# this method is not currently used, but it could also be too prospectively usefull to delete.
# TODO move to a utils file.
'''
def get_centres(ga_out):

    Do optimization, but then get cluster centres of parameters
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
    est = KMeans(n_clusters=1)
    est.fit(X)
    y_kmeans = est.predict(X)
    centers = est.cluster_centers_
    return td, test_opt, centres
'''

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


def ugly_score_wrangler(model,objectives2,to_latex_string=False):
    strict_scores = {}


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    for i in objectives2:
        test = i.features[0].test
        try:
            test.observation['value'] = np.abs(float(test.observation['mean']))*pq.dimensionless
            test.observation['n'] = 1
        except:
            temp = {}
            temp['value'] = np.abs(test.observation['value'])*pq.dimensionless
            temp['n'] = 1
            test.observation = temp

        try:
            test.prediction['value'] = np.abs(float(test.prediction))*pq.dimensionless
        except:
            temp = {}
            try:
                temp['value'] = np.abs(float(test.prediction))*pq.dimensionless#*pq.dimensionless
            except:
                #print(test.prediction.keys())
                try:
                    temp['value'] = np.abs(float(test.prediction['mean']))*pq.dimensionless#*pq.dimensionless
                except:
                    temp['value'] = np.abs(float(test.prediction['value']))*pq.dimensionless#*pq.dimensionless
                #temp['std'] = temp['mean']#*pq.dimensionless
            temp['n'] = 1
            test.prediction = temp

        test.score_type = RatioScore
        for k,v in test.observation.items():
            test.observation[k] = np.abs(test.observation[k]) #* test.observation[k]#.units

        for k,v in test.prediction.items():
            test.prediction[k] = np.abs(test.prediction[k])# * test.prediction[k]#.units

        re_score = test.compute_score(test.observation,test.prediction)
        strict_scores[test.name] = re_score
        #print(t.name)
    df = pd.DataFrame([strict_scores])

    df = df.T
    return df
'''
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
'''
#from neuronunit.tests import RheobaseTest, RheobaseTestP
def get_rh(dtc,rtest_class):
    '''
    :param dtc:
    :param rtest_class:
    :return:
    This is used to generate a rheobase test, given unknown experimental observations.s
    '''
    place_holder = {'n': 86, 'mean': 10 * pq.pA, 'std': 10 * pq.pA, 'value': 10 * pq.pA}
    backend_ = dtc.backend
    rtest = RheobaseTest(observation=place_holder,
                        name='RheobaseTest')

    dtc.rheobase = None
    assert len(dtc.attrs)
    model = dtc.dtc_to_model()
    #model.set_attrs(**dtc.attrs)
    rtest.params['injected_square_current'] = {}
    rtest.params['injected_square_current']['delay'] = DELAY
    rtest.params['injected_square_current']['duration'] = DURATION

    dtc.rheobase = rtest.generate_prediction(model)['value']
    temp_vm = model.get_membrane_potential()
    if np.isnan(np.min(temp_vm)):
        print('sampled nan')
        # rheobase exists but the waveform is nuts.
        # this is the fastest way to filter out a gene
        dtc.rheobase = None
    return dtc


#def substitute_parallel_for_serial(rtest):
#    rtest = RheobaseTestP(rtest.observation)
#    return rtest

def is_parallel_rheobase_compatible(backend):
    incompatible_backends = ['IZHI', 'HH', 'ADEXP']

    incompatible = any([x in backend for x in incompatible_backends])
    return not incompatible

def get_new_rtest(dtc):
    place_holder = {'n': 86, 'mean': 10 * pq.pA, 'std': 10 * pq.pA, 'value': 10 * pq.pA}
    #f = RheobaseTestP if is_parallel_rheobase_compatible(dtc.backend) else RheobaseTest
    f = RheobaseTest
    # if is_parallel_rheobase_compatible(dtc.backend) else RheobaseTest

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
            #if is_parallel_rheobase_compatible(dtc.backend):
            #    rtest = substitute_parallel_for_serial(rtest)
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
    if not hasattr(dtc,'SA'):
        dtc.SA = None


    if hasattr(dtc,'tests'):
        if type(dtc.tests) is type({}) and str('RheobaseTest') in dtc.tests.keys():
            rtest = dtc.tests['RheobaseTest']
            #if str('JHH') in dtc.backend or str('BHH') in dtc.backend or str("GLIF") in dtc.backend:# or str("HH") in dtc.backend:
                 #rtest = RheobaseTestP(dtc.tests['RheobaseTest'].observation)
                 #rtest.params = dtc.tests['RheobaseTest'].params
        else:
            rtest = get_rtest(dtc)
    else:
        rtest = get_rtest(dtc)

    if rtest is not None:
        model = dtc.dtc_to_model()
        if dtc.attrs is not None:
            model.attrs = dtc.attrs
        if isinstance(rtest,Iterable):
            rtest = rtest[0]
        dtc.rheobase = rtest.generate_prediction(model)['value']
        #print(dtc.rheobase)
        temp_vm = model.get_membrane_potential()
        min = np.min(temp_vm)
        if np.isnan(temp_vm.any()):
            print('sampled nan')

            dtc.rheobase = None
            # rheobase does exist but lets filter out this bad gene.
        return dtc
    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
    return dtc


'''
def mint_izhi_NEURON_model(dtc):
    """
    Depricated.
    """
    LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
    pre_model.model_path = LEMS_MODEL_PATH
    pre_model = dtc_to_rheo(pre_model)

    from neuronunit.models.reduced import ReducedModel#, VeryReducedModel
    model = ReducedModel(dtc.model_path,name='vanilla', backend=(dtc.backend, {'DTC':dtc}))
    pre_model.current_src_name = model._backend.current_src_name
    assert type(pre_model.current_src_name) is not type(None)
    dtc.cell_name = model._backend.cell_name
    model.attrs = pre_model.attrs
    return model
'''
import plotly.express as px
import time
def timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        print('time taken on block {0} '.format(t2-t1))
        return f
    return inner
#@timer
def inject_and_plot_model(pre_model,figname=None,plotly=True, verbose=False):
    # get rheobase injection value
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    #print(pre_model.attrs,'attrs in before plot \n\n\n\n')

    pre_model = dtc_to_rheo(pre_model)
    #print(pre_model.attrs,'attrs in after b plot \n\n\n\n')

    model = pre_model.dtc_to_model()
    #print(pre_model.attrs,'attrs in after a plot \n\n\n\n')

    uc = {'amplitude':pre_model.rheobase,'duration':DURATION,'delay':DELAY}

    if pre_model.jithub or "NEURON" in str(pre_model.backend):
        #if "JIT_" in model.backend:
        vm = model._backend.inject_square_current(**uc)
    else:
        vm = model.inject_square_current(uc)

    #if (str(pre_model.backend) in "NEURON" and not str(pre_model.backend) in "NEURONHH") or str(pre_model.backend) in "MAT" :
    #    vm = model._backend.inject_square_current(uc)
    vm = model.get_membrane_potential()
    if verbose:
        if vm is not None:
            print(vm[-1],vm[-1]<0*pq.mV)
    if vm is None:
        return None,None,None


    if not plotly:
        plt.clf()
        plt.figure()
        if pre_model.backend in str("HH"):
            plt.title('Conductance based model membrane potential plot')
        else:
            plt.title('Membrane potential plot')
        plt.plot(vm.times, vm.magnitude, 'k')
        plt.ylabel('V (mV)')
        plt.xlabel('Time (s)')

        if figname is not None:
            plt.savefig(str(figname)+str('.png'))
        plt.plot(vm.times,vm.magnitude)

    if plotly:
        fig = px.line(x=vm.times, y=vm.magnitude, labels={'x':'t (s)', 'y':'V (mV)'})
        if figname is not None:
            fig.write_image(str(figname)+str('.png'))
        else:
            return vm,fig
    return vm,plt,pre_model

def inject_and_plot_passive_model(pre_model,second=None,figname=None,plotly=True):


    # get rheobase injection value
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    #pre_model = dtc_to_rheo(pre_model)
    model = pre_model.dtc_to_model()
    uc = {'amplitude':-10*pq.pA,'duration':500*pq.ms,'delay':100*pq.ms}
    model.inject_square_current(uc)
    vm = model.get_membrane_potential()


    if second is not None:
        model2 = second.dtc_to_model()
        uc = {'amplitude':-10*pq.pA,'duration':500*pq.ms,'delay':100*pq.ms}
        model2.inject_square_current(uc)
        vm2 = model2.get_membrane_potential()
    if plotly and second is None:
        fig = px.line(x=vm.times, y=vm.magnitude, labels={'x':'t (ms)', 'y':'V (mV)'})
        #if figname is not None:
        #    fig.write_image(str(figname)+str('.png'))
        #else:
        return fig
    if plotly and second is not None:
        #model = second.dtc_to_model()
        #uc = {'amplitude':-10*pq.pA,'duration':500*pq.ms,'delay':100*pq.ms}
        #model.inject_square_current(uc)
        #vm1 = model.get_membrane_potential()
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm.times[0:-1]], y=[float(i) for i in vm.magnitude[0:-1]], name="yaxis data"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm2.times[0:-1]], y=[float(i) for i in vm2.magnitude[0:-1]], name="yaxis2 data"),
            secondary_y=True,
        )
        # Add figure title
        fig.update_layout(
            title_text="Compare traces"
        )
        # Set x-axis title
        fig.update_xaxes(title_text="time (ms)")
        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 1", secondary_y=False)
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 2", secondary_y=True)
        return fig
        #if figname is not None:
        #    fig.write_image(str(figname)+str('.png'))
        #else:
        #    fig.show()
    if not plotly:
        matplotlib.rcParams.update({'font.size': 15})
        plt.figure()
        if pre_model.backend in str("HH"):
            plt.title('Hodgkin-Huxley Neuron')
        else:
            plt.title('Membrane Potential')
        plt.plot(vm.times, vm.magnitude, c='b')#+str(model.attrs['a']))

        plt.plot(vm2.times, vm2.magnitude, c='g')
        plt.ylabel('Time (sec)')

        plt.ylabel('V (mV)')
        plt.legend(loc="upper left")

        if figname is not None:
            plt.savefig('thesis_simulated_data_match.png')
    #plt.plot(vm.times,vm.magnitude)

    return vm,plt

def inject_and_not_plot_model(pre_model,known_rh=None):

    # get rheobase injection value
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    model = pre_model.dtc_to_model()

    if known_rh is None:
        pre_model = dtc_to_rheo(pre_model)
        if type(model.rheobase) is type(dict()):
            uc = {'amplitude':model.rheobase['value'],'duration':DURATION,'delay':DELAY}
        else:
            uc = {'amplitude':model.rheobase,'duration':DURATION,'delay':DELAY}

    else:
        if type(known_rh) is type(dict()):
            uc = {'amplitude':known_rh['value'],'duration':DURATION,'delay':DELAY}
        else:
            uc = {'amplitude':known_rh,'duration':DURATION,'delay':DELAY}
    model.inject_square_current(uc)

    vm = model.get_membrane_potential()

    return vm
import plotly.graph_objects as go
from neuronunit.capabilities.spike_functions import get_spike_waveforms

def plotly_version(vm0,vm1,figname=None,snippets=False):

    import plotly.graph_objects as go
    if snippets:
        snippets1 = get_spike_waveforms(vm1)
        snippets0 = get_spike_waveforms(vm0)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Scatter(x=[float(i) for i in snippets0.times[0:-1]], y=[float(i) for i in snippets0.magnitude[0:-1]], name="yaxis data"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=[float(i) for i in snippets1.times[0:-1]], y=[float(i) for i in snippets1.magnitude[0:-1]], name="yaxis2 data"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text="Double Y Axis Example"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="xaxis title")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
        fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

        fig.show()
        if figname is not None:
            fig.write_image(str(figname)+str('.png'))
        else:
            fig.show()

    else:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm0.times[0:-1]], y=[float(i) for i in vm0.magnitude[0:-1]], name="yaxis data"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm1.times[0:-1]], y=[float(i) for i in vm1.magnitude[0:-1]], name="yaxis2 data"),
            secondary_y=True,
        )

        # Add figure title
        #fig.update_layout(
        #    title_text="Double Y Axis Example"
        #)

        # Set x-axis title
        fig.update_xaxes(title_text="xaxis title")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 1", secondary_y=False)
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 2", secondary_y=True)

        if figname is not None:
            fig.write_image(str(figname)+str('.png'))
        else:
            fig.show()
def basic_expVar(trace1, trace2):
    # https://github.com/AllenInstitute/GLIF_Teeter_et_al_2018/blob/master/query_biophys/query_biophys_expVar.py
    '''This is the fundamental calculation that is used in all different types of explained variation.
    At a basic level, the explained variance is calculated between two traces.  These traces can be PSTH's
    or single spike trains that have been convolved with a kernel (in this case always a Gaussian)
    Input:
        trace 1 & 2:  1D numpy array containing values of the trace.  (This function requires numpy array
                        to ensure that this is not a multidemensional list.)
    Returns:
        expVar:  float value of explained variance
    '''

    var_trace1=np.var(trace1)
    var_trace2=np.var(trace2)
    var_trace1_minus_trace2=np.var(trace1-trace2)

    if var_trace1_minus_trace2 == 0.0:
        return 1.0
    else:
        return (var_trace1+var_trace2-var_trace1_minus_trace2)/(var_trace1+var_trace2)


def model_trace(pre_model):
    from neuronunit.tests.base import AMPL, DELAY, DURATION

    # get rheobase injection value
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    pre_model = dtc_to_rheo(pre_model)
    model = pre_model.dtc_to_model()
    uc = {'amplitude':model.rheobase,'duration':DURATION,'delay':DELAY}
    model.inject_square_current(uc)
    vm = model.get_membrane_potential()
    return vm
def check_binary_match(dtc0,dtc1,figname=None,snippets=True,plotly=True):

    vm0 = model_trace(dtc0)
    vm1 = model_trace(dtc1)

    if plotly:
        plotly_version(vm0,vm1,figname,snippets)
    else:
        matplotlib.rcParams.update({'font.size': 8})

        plt.figure()

        if snippets:
            plt.figure()

            snippets1 = get_spike_waveforms(vm1)
            snippets0 = get_spike_waveforms(vm0)
            plt.plot(snippets0.times,snippets0.magnitude,label=str('model type: '))#+label)#,label='ground truth')
            plt.plot(snippets1.times,snippets1.magnitude,label=str('model type: '))#+label)#,label='ground truth')
            if dtc0.backend in str("HH"):
                plt.title('Check for waveform Alignment variance exp: {0}'.format(basic_expVar(snippets1, snippets0)))
            else:
                plt.title('membrane potential: variance exp: {0}'.format(basic_expVar(snippets1, snippets0)))
            plt.ylabel('V (mV)')
            plt.legend(loc="upper left")

            if figname is not None:
                plt.savefig(figname)

        else:
            if dtc0.backend in str("HH"):
                plt.title('Check for waveform Alignment')
            else:
                plt.title('membrane potential plot')
            plt.plot(vm0.times, vm0.magnitude,label="target")
            plt.plot(vm1.times, vm1.magnitude,label="solutions")
            plt.ylabel('V (mV)')
            plt.legend(loc="upper left")

            if figname is not None:
                plt.savefig(figname)

from neuronunit.capabilities.spike_functions import get_spike_waveforms
def contrast(dtc0,dtc1,figname=None,snippets=True):
    matplotlib.rcParams.update({'font.size': 10})

    vm0 =inject_and_not_plot_model(dtc0)
    vm1 =inject_and_not_plot_model(dtc1)


    if snippets:
        snippets_ = get_spike_waveforms(vm)
        dtc.snippets = snippets_
        plt.plot(snippets_.times,snippets_,color=color,label=str('model type: ')+label)#,label='ground truth')
    else:
        plt.plot(vm.times,vm,color=color,label=str('model type: ')+label)#,label='ground truth')
    ax.legend()


    plt.figure()
    if dtc0.backend in str("HH"):
        plt.title('Check for waveform Alignment')
    else:
        plt.title('membrane potential plot')
    plt.plot(vm0.times, vm0.magnitude,label="best solution")
    plt.plot(vm1.times, vm1.magnitude,label="worst solution")
    plt.ylabel('V (mV)')
    plt.legend(loc="upper left")
    if figname is not None:
        plt.savefig(figname)

    return plt


def check_match_front(dtc0,dtcpop,figname = None):
    matplotlib.rcParams.update({'font.size': 12})

    vm0 =inject_and_not_plot_model(dtc0)

    vms = []
    for dtc in dtcpop:
        vms.append(inject_and_not_plot_model(dtc))

    plt.figure()
    if dtc0.backend in str("HH"):
        plt.title('Check for waveform Alignment')
    else:
        plt.title('membrane potential plot')
    plt.plot(vm0.times, vm0.magnitude,label="target",c='red')
    plt.plot(vms[0].times, vms[0].magnitude,label="best candidate",c='blue', linewidth=4)

    for v in vms:
        plt.plot(v.times, v.magnitude,c='grey')
    plt.ylabel('V (mV)')
    plt.legend(loc="upper left")
    if figname is not None:
        plt.savefig(figname)
    return plt


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

def jrt(use_test,backend,protocol={'elephant':True,'allen':False}):
    from neuronunit.optimization import model_parameters
    use_test = TSD(use_test)
    use_test.use_rheobase_score = True
    edges = model_parameters.MODEL_PARAMS[backend]
    #if protocol is 'elephant':
    OM = OptMan(use_test,
        backend=backend,
        boundary_dict=edges,
        protocol=protocol)
    '''
    else:
        OM = OptMan(use_test,
            backend=backend,
            boundary_dict=edges,
            protocol={'allen': True, 'elephant': False})
    '''
    return OM

def switch_logic(xtests):
    # move this logic into sciunit tests
    '''
    Hopefuly depreciated by future NU debugging.
    '''
    try:
        aTSD = TSD()
    except:
        #basically an object defined in the currently inhabited file:
        # mystery why above won't work.
        aTSD = neuronunit.optimization.optimization_management.TSD()

    if type(xtests) is type(aTSD):
        xtests = list(xtests.values())
    if type(xtests) is type(list()):
        pass
    for t in xtests:
        if str('RheobaseTest') == t.name:
            t.active = True
            t.passive = False
        #elif str('RheobaseTestP') == t.name:
        #    t.active = True
        #    t.passive = False
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
            t.passive = False
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
        if isinstance(rheobase,type(dict())):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase
    return keyed



def passive_values(keyed):
    PASSIVE_DURATION = 500.0*pq.ms
    PASSIVE_DELAY = 200.0*pq.ms
    keyed['injected_square_current'] = {}
    keyed['injected_square_current']['delay']= PASSIVE_DELAY
    keyed['injected_square_current']['duration'] = PASSIVE_DURATION
    keyed['injected_square_current']['amplitude'] = -10*pq.pA
    return keyed

def neutral_values(keyed):
    PASSIVE_DURATION = 500.0*pq.ms
    PASSIVE_DELAY = 200.0*pq.ms
    keyed['injected_square_current'] = {}
    keyed['injected_square_current']['delay']= PASSIVE_DELAY
    keyed['injected_square_current']['duration'] = PASSIVE_DURATION
    keyed['injected_square_current']['amplitude'] = 0*pq.pA
    return keyed



'''

def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))

import copy
def get_dtc_pop(contains_dtcpop,filtered_tests,model_parameters,backend = 'ADEXP'):
    from neuronunit.optimization.optimizations import SciUnitoptimization

    random.seed(64)
    boundary_dict = model_parameters.MODEL_PARAMS[backend]
    tests=filtered_tests['Hippocampus CA1 basket cell']
    try:
        DO = SciUnitoptimization(offspring_size = 1,
            error_criterion = tests, boundary_dict = boundary_dict,
                                     backend = backend)
    except:
        DO = SciUnitoptimization(offspring_size = 1,
            error_criterion = tests, boundary_dict = boundary_dict,
                                     backend = backend)

    DO.setnparams(nparams = len(contains), boundary_dict = boundary_dict)
    DO.setup_deap()

    dtcdic = {}
    for k,v in contains_dtcpop[backend].items():

        dtcpop = []
        for i in v:
            dtcpop.append(transform((i,DO.td,backend)))
            dtcpop[-1] = dask.compute(dtcpop[-1])[0]
            dtcpop[-1].backend = backend
            dtcpop[-1] = DO.OptMan.dtc_to_rheo(dtcpop[-1])
            dtcpop[-1] = DO.OptMan.format_test(dtcpop[-1])

        dtcdic[k] = copy.copy(dtcpop)
    return dtcdic, DO

def allen_wave_predictions(dtc,thirty=False):

    if thirty:
        vm30 = dtc.vm30
    else:
        vm30 = dtc.vm_soma

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

        rectify unfilterable high sample frequencies by downsampling them
        downsample too densely sampled signals.
        Making them amenable to Allen analysis
        v = [ float(v*1000.0) for v in vm30.magnitude]
        t = [ float(t) for t in vm30.times ]
        ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
        ephys.process_spikes()

    ephys_dict = ephys.as_dict()
    if not 'spikes' in ephys_dict.keys() or ephys_dict['avg_rate'] == 0:
        dtc.scores = None
        dtc.preds = {}
        return dtc, {}
    else:
        prediction = {}
        dtc.preds= {}
        obs= {}
        for k in ephys_dict.keys():
            if 'spikes' not in k:
                dtc.preds[k] = ephys_dict[k]

            else:

                for wavef in ephys_dict['spikes'][0].keys():
                    temp = ephys_dict['spikes'][0][wavef]
                    dtc.preds[wavef+str('_first')] = temp
                for wavef in ephys_dict['spikes'][-1].keys():
                    temp = ephys_dict['spikes'][-1][wavef]
                    dtc.preds[wavef+str('_last')] = temp
                half = int(len(ephys_dict['spikes'])/2.0)
                for wavef in ephys_dict['spikes'][half].keys():
                    temp = ephys_dict['spikes'][half][wavef]
                    dtc.preds[wavef+str('_half')] = temp
                dtc.spike_cnt = len(ephys_dict['spikes'])
                dtc.preds['spikes'] = dtc.spike_cnt
        if thirty:
            dtc.allen_30 = dtc.preds
        else:
            dtc.allen_15 = dtc.preds
        del vm30
        return dtc,ephys
from scipy.interpolate import interp1d

# from scipy.interpolate import interp1d
'''
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


        dtc.spike_cnt = len(ephys_dict['spikes'])
        dtc.preds['spikes'] = dtc.spike_cnt
    return dtc


'''
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
        # rectify unfilterable high sample frequencies by downsampling them
        # downsample too densely sampled signals.
        # Making them amenable to Allen analysis

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
            # garuntee a maximum number of aligned reference indexs in disparate length spike trains.
            garunteed_reference_points = downsample(list(range(0,len(ephys_dict['spikes'])), npts))
        dtc = append_spikes(ephys_dict,dtc)
        return dtc,compare,ephys
'''

'''
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
    score_type = scores.ZScore
    helper.score_type = score_type
    dtc.preds = {}
    dtc.tests = {}

    for k,observation in compare.items():
        if  str('spikes') not in k:
            # compute interspike firing frame_dynamics

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

        # compute perspike waveform features on just the first spike
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


                prediction = {'mean': ephys_dict['spikes'][0][key]}
                helper.name = str(key)
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
    '''
def new_model(dtc):
    model = dtc.dtc_to_model()
    model.set_attrs(dtc.attrs)
    return model

'''
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
    stimulus = [ s for s in ephys_sweeps if s['stimulus_units'] == 'Amps' \
     if s['num_spikes'] != None \
     if s['stimulus_name']!='Ramp' and s['stimulus_name']!='Short Square']
    amplitudes = [ s['stimulus_absolute_amplitude'] for s in stimulus ]
    durations = [ s['stimulus_duration'] for s in stimulus ]
    expeceted_spikes = [ s['num_spikes'] for s in stimulus ]
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
'''




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
    model.vm_soma = None
    model.vm_soma = dtc.vm_soma
    if type(dtc.rheobase) is type(dict()):
        rheobase = dtc.rheobase['value']
    else:
        rheobase = dtc.rheobase
    model.druckmann2013_standard_current = None
    model.druckmann2013_standard_current = rheobase * 1.5
    model.vm30 = None
    model.vm30 = dtc.vm30
    if rheobase <0.0 or np.max(dtc.vm30)<0.0:# or model.get_spike_count()<1:
        dtc.dm_test_features = None
        return dtc
    model.druckmann2013_strong_current = None
    model.druckmann2013_strong_current = rheobase * 3.0

    model.druckmann2013_input_resistance_currents =[ -5.0*pq.pA, -10.0*pq.pA, -15.0*pq.pA]#,copy.copy(current)
    from neuronunit.tests import dm_test_container #import Interoperabe

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
    model.set_attrs(dtc.attrs)
    try:
        values = [v for v in dtc.protocols.values()][0]

    except:
        values = [v for v in dtc.tests.values()][0]
    current = values['injected_square_current']
    current['amplitude'] = dtc.rheobase * 1.5
    model.inject_square_current(current)
    vm15 = model.get_membrane_potential()
    model.vm_soma = None
    model.vm_soma = vm15
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
'''
def nuunit_dm_rheo_evaluation(dtc):
    model = dtc.dtc_to_model()
    model.set_attrs(dtc.attrs)
    try:
        values = [v for v in dtc.protocols.values()][0]
    except:
        values = [v for v in dtc.tests.values()][0]

    current = values['injected_square_current']
    current['amplitude'] = dtc.rheobase
    model.inject_square_current(current)
    vm15 = model.get_membrane_potential()
    model.vm_soma = None
    model.vm_soma = vm15
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
'''
#from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

def train_length(dtc):
    if not hasattr(dtc,'everything'):
        dtc.everything = {}
    vm = copy.copy(dtc.vm_soma)
    train_len = float(len(sf.get_spike_train(vm)))
    dtc.everything['Spikecount_1.5x'] = copy.copy(train_len)
    return dtc

def three_step_protocol(dtc,solve_for_current=None):
    if solve_for_current is None:
        _,_,_,_,dtc = inject_model_soma(dtc)
        if dtc.vm_soma is None:
            return dtc
        try:
            dtc = efel_evaluation(dtc,thirty=False)
        except:
            dtc.vm_soma = None
    else:
        _,_,_,dtc = inject_model_soma(dtc,solve_for_current=solve_for_current)
        if dtc.vm_soma is None:
            return dtc
        dtc = efel_evaluation(dtc,thirty=False)

    dtc = rekeyed(dtc)
    if dtc.everything is not None:
        dtc = train_length(dtc)
    return dtc

    #if hasattr(dtc,'allen_30'):
    #    dtc = rekeyed(dtc)
    #    if hasattr(dtc,'dm_test_features'):
    #        dtc.everything.update(dtc.dm_test_features)

'''
def retestobs(dtc):
    if type(dtc.tests) is not type(list()):
        dtc.tests = list(dtc.tests.values())
    for t in dtc.tests:
        t.observation['std'] = np.abs(t.observation['std'])
    return dtc

def rekeyeddm(dtc):
    standard = 0
    strong = 0
    easy_map = [
                {'AP12AmplitudeDropTest':standard},
                {'AP1SSAmplitudeChangeTest':standard},
                {'AP1AmplitudeTest':standard},
                {'AP1WidthHalfHeightTest':standard},
                {'AP1WidthPeakToTroughTest':standard},
                {'AP1RateOfChangePeakToTroughTest':standard},
                {'AP1AHPDepthTest':standard},
                {'AP2AmplitudeTest':standard},
                {'AP2WidthHalfHeightTest':standard},
                {'AP2WidthPeakToTroughTest':standard},
                {'AP2RateOfChangePeakToTroughTest':standard},
                {'AP2AHPDepthTest':standard},
                {'AP12AmplitudeChangePercentTest':standard},
                {'AP12HalfWidthChangePercentTest':standard},
                {'AP12RateOfChangePeakToTroughPercentChangeTest':standard},
                {'AP12AHPDepthPercentChangeTest':standard},
                {'InputResistanceTest':str('ir_currents')},
                {'AP1DelayMeanTest':standard},
                {'AP1DelaySDTest':standard},
                {'AP2DelayMeanTest':standard},
                {'AP2DelaySDTest':standard},
                {'Burst1ISIMeanTest':standard},
                {'Burst1ISISDTest':standard},
                {'InitialAccommodationMeanTest':standard},
                {'SSAccommodationMeanTest':standard},
                {'AccommodationRateToSSTest':standard},
                {'AccommodationAtSSMeanTest':standard},
                {'AccommodationRateMeanAtSSTest':standard},
                {'ISICVTest':standard},
                {'ISIMedianTest':standard},
                {'ISIBurstMeanChangeTest':standard},
                {'SpikeRateStrongStimTest':strong},
                {'AP1DelayMeanStrongStimTest':strong},
                {'AP1DelaySDStrongStimTest':strong},
                {'AP2DelayMeanStrongStimTest':strong},
                {'AP2DelaySDStrongStimTest':strong},
                {'Burst1ISIMeanStrongStimTest':strong},
                {'Burst1ISISDStrongStimTest':strong},
            ]
    dm_labels = [list(keys.keys())[0] for keys in easy_map ]
    rekeyed = {}
    dmtf = dtc.dm_test_features
    keep_columns = {}
    for l in easy_map:
        for k in l.keys():
            if str(k)+str('_3.0x') in dmtf.keys():
                keep_columns.append(str(k)+str('_3.0x'))
            elif str(k)+str('_1.5x') in df.columns:
                keep_columns.append(str(k)+str('_1.5x'))
    return dtc
'''
def rekeyed(dtc):
    rekey = {}
    if hasattr(dtc,'allen_30'):
        for k,v in dtc.allen_30.items():
            rekey[str(k)+str('_3.0x')] = v
    if hasattr(dtc,'allen_15'):
        for k,v in dtc.allen_15.items():
            rekey[str(k)+str('_1.5x')] = v
    if hasattr(dtc,'efel_30'):
        for k,v in dtc.efel_30[0].items():
            rekey[str(k)+str('_3.0x')] = v
    if hasattr(dtc,'efel_15'):
        if dtc.efel_15 is not None:
            for k,v in dtc.efel_15[0].items():
                rekey[str(k)+str('_1.5x')] = v
        else:
            rekey = None
    dtc.everything = rekey
    return dtc
'''
from neuronunit.tests.fi import RheobaseTest
import quantities as pq

import scipy.stats as scs
from scipy.stats import norm

def z_val(sig_level=0.05, two_tailed=True):
    """Returns the z value for a given significance level"""
    z_dist = scs.norm()
    if two_tailed:
        sig_level = sig_level/2
        area = 1 - sig_level
    else:
        area = 1 - sig_level

    z = z_dist.ppf(area)

    return z
'''

def initialise_test(v,rheobase):
    v = switch_logic([v])
    v = v[0]
    k = v.name
    if not hasattr(v,'params'):
        v.params = {}
    if not 'injected_square_current' in v.params.keys():
        v.params['injected_square_current'] = {}
    if v.passive == False and v.active == True:
        keyed = v.params['injected_square_current']
        v.params = active_values(keyed,rheobase)
        v.params['injected_square_current']['delay'] = DELAY
        v.params['injected_square_current']['duration'] = DURATION
    if v.passive == True and v.active == False:

        v.params['injected_square_current']['amplitude'] =  -10*pq.pA
        v.params['injected_square_current']['delay'] = PASSIVE_DELAY
        v.params['injected_square_current']['duration'] = PASSIVE_DURATION

    if v.name in str('RestingPotentialTest'):
        v.params['injected_square_current']['delay'] = PASSIVE_DELAY
        v.params['injected_square_current']['duration'] = PASSIVE_DURATION
        v.params['injected_square_current']['amplitude'] = 0.0*pq.pA

    return v


def make_allen_tests():

  rt = RheobaseTest(observation={'mean':70*qt.pA,'std':70*qt.pA})
  #tc = TimeConstantTest(observation={'mean':24.4*qt.ms,'std':24.4*qt.ms})
  ir = InputResistanceTest(observation={'mean':132*qt.MOhm,'std':132*qt.MOhm})
  rp = RestingPotentialTest(observation={'mean':-71.6*qt.mV,'std':77.5*qt.mV})

  allen_tests = [rt,rp,ir]
  for t in allen_tests:
      t.score_type = RatioScore
  allen_tests[-1].score_type = ZScore
  allen_suite482493761 = TestSuite(allen_tests)
  allen_suite482493761.name = "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/482493761"

  rt = RheobaseTest(observation={'mean':190*qt.pA,'std':190*qt.pA})
  #tc = TimeConstantTest(observation={'mean':13.8*qt.ms,'std':13.8*qt.ms})
  ir = InputResistanceTest(observation={'mean':132*qt.MOhm,'std':132*qt.MOhm})
  rp = RestingPotentialTest(observation={'mean':-77.5*qt.mV,'std':77.5*qt.mV})

  allen_tests = [rt,rp,ir]
  for t in allen_tests:
      t.score_type = RatioScore
  allen_tests[-1].score_type = ZScore
  allen_suite471819401 = TestSuite(allen_tests)
  allen_suite471819401.name = "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/471819401"
  list_of_dicts = []
  cells={}
  cells['471819401'] = TSD(allen_suite471819401)
  cells['482493761'] = TSD(allen_suite482493761)

  for k,v in cells.items():
      observations = {}
      for k1 in cells['482493761'].keys():
          vsd = TSD(v)
          if k1 in vsd.keys():
              vsd[k1].observation['mean']

              observations[k1] = np.round(vsd[k1].observation['mean'],2)
              observations['name'] = k
      list_of_dicts.append(observations)
  df = pd.DataFrame(list_of_dicts)
  return allen_suite471819401,allen_suite482493761,df


def constrain_ahp(vm_used,rheobase):
    #vm_used = inject_and_not_plot_model(dtc)
    efel.reset()
    efel.setThreshold(0)
    trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
          'V': [float(v) for v in vm_used.magnitude]}#,
    #          'stimulus_current': [float(rheobase)]}
    DURATION = 1100*qt.ms
    DELAY = 100*qt.ms
    trace3['stim_end'] = [ float(DELAY)+float(DURATION) ]
    trace3['stim_start'] = [ float(DELAY)]
    simple_yes_list = [
    'AHP_depth',
    'AHP_depth_abs',
    'AHP_depth_last','all_ISI_values','ISI_values', 'time_to_first_spike',
    'time_to_last_spike',
    'time_to_second_spike',
    'trace_check']

    results = efel.getMeanFeatureValues([trace3],simple_yes_list)#, parallel_map=pool.map)
    return results

class NUFeature_standard_suite(object):
    def __init__(self,test,model):
        self.test = test
        self.model = model
    def calculate_score(self,responses):
        if 'model' in responses.keys():
            dtc = responses['model']
            model = responses['model'].dtc_to_model()
        else:
            return 100.0
        if type(responses['response']) is type(list()):
            return 1000.0

        if responses['response'] is not None:
            vm = responses['response']
            results = constrain_ahp(vm,dtc.rheobase)
            results = results[0]
            if results['AHP_depth'] is None or np.abs(results['AHP_depth_abs'])>=80:
                return 1000.0
            if np.abs(results['AHP_depth'])>=105:
                return 1000.0

            if np.max(vm)>=0:
                snippets = get_spike_waveforms(vm)
                widths = spikes2widths(snippets)
                #if isinstace(type(widths),type(Iterable)):
                try:
                    widths = widths[0]

                except:
                    pass

                spike_train = threshold_detection(vm, threshold=0*pq.mV)

                if (spike_train[0]+ 2.5*qt.ms) > vm.times[-1]:
                    too_long = True
                    return 1000.0

                if widths >= 2.0*qt.ms:
                    return 1000.0
                if float(vm[-1])==np.nan or np.isnan(vm[-1]):
                    return 1000.0
                if float(vm[-1])>=0.0:
                    return 1000.0

                assert vm[-1]<0*pq.mV
        model.attrs = responses['params']
        if 'rheobase' in responses.keys():
            self.test = initialise_test(self.test,responses['rheobase'])
        if "RheobaseTest" in str(self.test.name):
            self.test.score_type = ZScore
            prediction = {'value':responses['rheobase']}
            score_gene = self.test.compute_score(self.test.observation,prediction)
            lns = np.abs(np.float(score_gene.raw))
            return lns
        else:
            prediction = self.test.generate_prediction(model)
            score_gene = self.test.judge(model)
            try:
                score_gene = self.test.judge(model)
            except:
                return 100.0

        if not isinstance(type(score_gene),type(None)):
            if not isinstance(score_gene,sciunit.scores.InsufficientDataScore):
                try:
                    if not isinstance(type(score_gene.raw),type(None)):
                        lns = np.abs(np.float(score_gene.raw))
                    else:
                        if not isinstance(type(score_gene.raw),type(None)):
                            # works 1/2 time that log_norm_score does not work
                            # more informative than nominal bad score 100
                            lns = np.abs(np.float(score_gene.raw))
                            # works 1/2 time that log_norm_score does not work
                            # more informative than nominal bad score 100

                except:
                    lns = 100
            else:
                lns = 100
        else:
            lns = 100
        if lns==np.inf or lns==np.nan:
            lns = 100
        return lns

def make_evaluator(nu_tests,
                    PARAMS,
                    experiment=str('Neocortex pyramidal cell layer 5-6'),
                    model=str('IZHI')):


    if type(nu_tests) is type(list()):
        nu_tests[0].score_type = ZScore
    if type(nu_tests) is type(dict()):
        if "RheobaseTest" in nu_tests.keys():
            nu_tests["RheobaseTest"].score_type = ZScore
        nu_tests = list(nu_tests.values())


    if model == "IZHI":
        simple_cell = model_classes.IzhiModel()
    if model == "MAT":
        simple_cell = model_classes.MATModel()
    if model == "ADEXP":
        simple_cell = model_classes.ADEXPModel()
    dtc = DataTC()
    dtc.backend = simple_cell.backend
    dtc._backend = simple_cell._backend
    simple_cell.params = PARAMS#simple_cell._backend.default_attrs
    simple_cell.NU = True
    if "L5PC" in model:
        nu_tests_ = l5pc_specific_modifications(nu_tests)
        nu_tests = list(nu_tests_.values())
        simple_cell.name = "L5PC"

    else:
        simple_cell.name = model+experiment
    objectives = []
    for tt in nu_tests:
        feature_name = tt.name
        ft = NUFeature_standard_suite(tt,simple_cell)
        objective = ephys.objectives.SingletonObjective(
            feature_name,
            ft)
        objectives.append(objective)
    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
    sweep_protocols = []
    protocol = ephys.protocols.SweepProtocol('step1', [None], [None])
    sweep_protocols.append(protocol)
    onestep_protocol = ephys.protocols.SequenceProtocol('onestep', protocols=sweep_protocols)
    cell_evaluator = ephys.evaluators.CellEvaluator(
            cell_model=simple_cell,
            param_names=copy.copy(BPO_PARAMS)[model].keys(),
            fitness_protocols={onestep_protocol.name: onestep_protocol},
            fitness_calculator=score_calc,
            sim='euler')
    simple_cell.params_by_names(copy.copy(BPO_PARAMS)[model].keys())
    print('called')
    return cell_evaluator, simple_cell, score_calc , [tt.name for tt in nu_tests]
import scipy

from multiprocessing import Process, Pipe
#from itertools import izip
import multiprocessing
# https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

'''
def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def pebble_parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]
'''
def get_binary_file_downloader_html(bin_file_path, file_label='File'):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file_path)}">Download {file_label}</a>'
    return href

def instance_opt(constraints,PARAMS,test_key,model_value,MU,NGEN,diversity,full_test_list=None,use_streamlit=True):
    import utils #as utils
    import bluepyopt as bpop

    if type(constraints) is not type(list()):
        constraints = list(constraints.values())
    cell_evaluator, simple_cell, score_calc, test_names = make_evaluator(
                                                          constraints,
                                                          PARAMS,
                                                          test_key,
                                                          model=model_value)
    model_type = str('_best_fit_')+str(model_value)+'_'+str(test_key)+'_.p'
    mut = 0.1
    cxp = 0.4
    #pebble_used = pebble.ProcessPool(max_workers=1, max_tasks=4, initializer=None, initargs=None)

    #if model_value is not "HH" and model_value is not "NEURONHH" and model_value is not "IZHI":
        #print('2 backend, parallel slow down circumnavigated',cell_model.backend)

    print(cell_evaluator)
    #import pdb
    #pdb.set_trace()

    optimization = bpop.optimizations.DEAPoptimization(
                evaluator=cell_evaluator,
                offspring_size = MU,
                map_function = map,
                selector_name=diversity,
                mutpb=mut,
                cxpb=cxp)

    final_pop, hall_of_fame, logs, hist = optimization.run(max_ngen=NGEN)

    best_ind = hall_of_fame[0]
    best_ind_dict = cell_evaluator.param_dict(best_ind)
    model = cell_evaluator.cell_model
    cell_evaluator.param_dict(best_ind)
    model.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}
    opt = model.model_to_dtc()
    opt.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}
    if type(constraints) is type(TSD()):
        constraints = list(constraints.values())
    if hasattr(constraints,'tests'):# is type(TestSuite):
        constraints = constraints.tests
    opt.tests = constraints
    passed = False
    from IPython.display import display
    if len(constraints)>1:
        try:
            opt.self_evaluate(tests=constraints)

            obs_preds = opt.make_pretty(constraints)
            display(obs_preds)
            passed = True
        except:
            #import pdb
            #pdb.set_trace()
            opt.self_evaluate(tests=constraints)
            print(opt.SA)
            display(pd.DataFrame(opt.SA))
            print('something went wrong with stats')
            #import pdb
            #pdb.set_trace()
            passed = False
    if passed:


        zvalues_opt = opt.SA.values
        chi_sqr_opt= np.sum(np.array(zvalues_opt)**2)
        p_value = 1-scipy.stats.chi2.cdf(chi_sqr_opt, 8)
        frame = opt.SA.to_frame()
        score_frame = frame.T
        obs_preds = opt.obs_preds.T
        #return final_pop, hall_of_fame, logs, hist, opt
    else:
        return final_pop, hall_of_fame, logs, hist, opt, None, None, None

    if not use_streamlit:
        return final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value

    else:

        import streamlit as st
        #opt.self_evaluate(tests=use_tests)

        #opt.self_evaluate(opt.tests)
        if full_test_list is not None:
            opt.make_pretty(full_test_list)
        else:
            if len(constraints)<len(opt.tests):
                use_tests = opt.tests
            else:
                use_tests = constraints
            opt.make_pretty(use_tests)

        st.markdown('---')
        st.success("Model best fit to experiment {0}".format(test_key))
        #st.markdown("Would you like to pickle the optimal model? (Note not implemented yet, but trivial)")

        st.markdown('---')
        #st.write(score_frame)

        st.markdown('\n\n\n\n')

        #obs_preds.rename(columns=)
        st.table(obs_preds)
        st.markdown('\n\n\n\n')

        #try:
        #  st.dataframe(obs_preds.style.background_gradient(cmap ='viridis').set_properties(**{'font-size': '20px'}))
        #except:

        #  sns.heatmap(obs_preds, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        #  st.pyplot()
        '''
        sns.set(context="paper", font="monospace")

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(12, 12))

        g = sns.heatmap(score_frame, linewidths = 0.30, annot = True)
        g.set_xticklabels(g.get_xticklabels(),rotation=45)

        st.pyplot()
        '''

        #st.markdown("""
        #-----
        #({0}, and {1}):
        #-----
        #""")

        st.markdown('\n\n\n\n')

        st.markdown("----")
        st.markdown("""
        -----
        The optimal model parameterization is
        -----
        """)
        best_params_frame = pd.DataFrame([opt.attrs])
        st.write(best_params_frame)





        st.markdown("----")

        st.markdown("Model behavior at rheobase current injection")

        vm,fig = inject_and_plot_model(opt,plotly=True)
        st.write(fig)
        st.markdown("----")

        st.markdown("Model behavior at -10pA current injection")
        fig = inject_and_plot_passive_model(opt,opt,plotly=True)
        st.write(fig)
        st.markdown("----")
        st.write("($\chi^{2}$ and $p-value$) =")
        #st.write("($\chi^{2}$, $p-value$)=")
        st.markdown("({0} , {1})".format(chi_sqr_opt, p_value))

        #plt.show()



        #st.markdown("""
        #-----
        #Model Performance Relative to fitting data {0}
        #-----
        #""".format(sum(best_ind.fitness.values)/(30*len(constraints))))
        # radio_value = st.sidebar.radtio("Target Number of Samples",[10,20,30])
        #st.markdown("""
        #-----
        #This score is {0} worst score is {1}
        #-----
        #""".format(sum(best_ind.fitness.values),30*len(constraints)))
        #plot_as_normal(opt)

        return final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value,best_params_frame

def surface_plot(opt,hist,figname):
    dim = len(opt.attrs.keys())


    fig_trip,ax_trip = plt.subplots(int(dim),int(dim),figsize=(20,20))

    flat = ax_trip.flatten()
    cnt =0
    for i,k0 in enumerate(simple_cell.attrs.keys()):
        for j,k1 in enumerate(simple_cell.attrs.keys()):
            if i<j:
                if k0!=k1:
                    from neuronunit.plottools import plot_surface2

                    ax_trip,plot_axis = plot_surface2(fig_trip,flat[cnt],k0,k1,hist,opt,opt)
        cnt+=1
    plt.legend()
    plt.savefig(figname)

def full_statistical_description(constraints,\
                                exp_cell,MODEL_PARAMS,test_key,\
                                model_value,MU,NGEN,diversity,\
                                full_test_list=None,use_streamlit=False,\
                                tf=None,dry_run=True):

    #buffer = TSD(constraints)
    #if "FITest" in constraints.keys():
    #    tests["FITest"] = constraints["FITest"]
    #constraints[exp_cell] = list(tests.values())
    if type(constraints) is type(list()):
        constraints = TSD(constraints)

    if hasattr(constraints,'keys'):
        keys = constraints.keys()

    else:
        #keys = []
        constraints_d = {}
        for t in constraints:
            constraints_d[t.name] = t
        constraints = constraints_d
        keys = constraints.keys()
            #keys.append(t.name)

    final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value = instance_opt(constraints,
            MODEL_PARAMS,test_key,model_value,MU,NGEN,diversity,full_test_list=keys,use_streamlit=False)
    #MODEL_PARAMS,"allen","GLIF",
    #MU,NGEN,"IBEA",'Neocortex pyramidal cell layer 5-6',use_streamlit=False)
    #surface_plot(opt,hist,figname=str(exp_cell)+str('_')+str(model_value)+'_.png')
    temp = final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value
    opt_pickle =  opt, obs_preds, chi_sqr_opt, p_value
    pickle.dump(opt_pickle,open(str(exp_cell)+str('_')+str(model_value)+'_.p',"wb"))

    gen_numbers = logs.select('gen')
    min_fitness = logs.select('min')
    max_fitness = logs.select('max')
    mean_fitness = logs.select('avg')

    plt.plot(gen_numbers, min_fitness, label='min fitness')
    plt.plot(gen_numbers, max_fitness, label='max fitness')
    plt.plot(gen_numbers, mean_fitness, label='mean fitness')
    plt.semilogy()

    plt.xlabel('generation #')

    plt.ylabel('score (# std)')
    plt.legend()
    plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    #plt.ylim(0.9*min(min_fitness), 1.1 * max(np.log(max_fitness)))
    plt.savefig(str('log_evolution_stats_')+str(exp_cell)+str('_')+str(model_value)+'_.png')
    plt.clf()
    plt.plot(gen_numbers, min_fitness, label='min fitness')
    plt.plot(gen_numbers, max_fitness, label='max fitness')
    plt.plot(gen_numbers, mean_fitness, label='mean fitness')
    #plt.semilogy()

    plt.xlabel('generation #')

    plt.ylabel('score (# std)')
    plt.legend()
    plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    #plt.ylim(0.9*min(min_fitness), 1.1 * max(np.log(max_fitness)))
    plt.savefig(str('no_log_evolution_stats_')+str(exp_cell)+str('_')+str(model_value)+'_.png')
    final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value = temp
    if not dry_run:
        if tf is None:
            tf = open(str(exp_cell)+str(model_value)+str('.tex'),'w')

        tf.write("\subsubsection{"+str(exp_cell)+" "+str(model_value)+"}")
        tf.write(pd.DataFrame([{'chi_square':chi_sqr_opt,'p_value':p_value}]).T.to_latex())
        best_params_frame = pd.DataFrame([opt.attrs])
        tf.write('optimal model parameters')
        tf.write(best_params_frame.to_latex())
        try:
            opt.make_pretty(opt.tests)

            df=opt.obs_preds
            tf.write(df.to_latex())
        except:
            #opt.make_pretty(opt.tests)

            #df=opt.obs_preds
            tf.write("failed model")
        os.system("cat "+str(exp_cell)+str(model_value)+str('.tex'))
    results_dict={}
    results_dict['model_value'] = model_value
    results_dict['exp_cell'] = exp_cell # = {}
    results_dict['chi_square'] = chi_sqr_opt
    results_dict['p_value'] = p_value
    df = pd.DataFrame([results_dict])

    return df, results_dict, opt


import os

def plot_as_normal(dtc):
    import streamlit as st

    #fig,axes = plt.subplots(2,math.ceil(len(dtc.tests)/2),figsize=(20,8))
    collect_z_offset = []
    for i,t in enumerate(dtc.tests):
        #ax = axes.flat[i]
        t.score_type = ZScore
        model = dtc.dtc_to_model()
        score = t.judge(model)
        x1 = -1.01
        x2 = 1.0
        sigma = 1.0
        mu = 0
        x = np.arange(-sigma, sigma, 0.001) # range of x in spec
        x_all = np.arange(-sigma, sigma, 0.001)
        y_point = norm.pdf(mu+float(score.raw),0,1)
        y2 = norm.pdf(x_all,0,1)

        y = norm.pdf(x,0,1)
        y2 = norm.pdf(x_all,0,1)



        x_point = mu+float(score.raw)
        collect_z_offset.append(score.raw)
        name = t.name.split('Test')[0]
        title = str(name)+str(' ')+str(t.observation['mean'].units)

        zplot(x_point,y_point,title)
        #break

        #ax.scatter(x_point,y_point,c='r',s=300,marker='o')
        #ax.plot(x_all,y2)
        #ax.set_xlim([-1.0,1.0])
        #ax.set_title()
        #fig.suptitle(str(name)+str(' ')+str(t.observation['mean'].units), fontsize=16)
        st.pyplot()
    #return np.mean(collect_z_offset)
def plot_as_normal_all(dtc,random):
    import streamlit as st

    #fig,axes = plt.subplots(2,math.ceil(len(dtc.tests)/2),figsize=(20,8))
    collect_z_offset = []
    collect_z_offset_random = []

    for i,t in enumerate(dtc.tests):
        #ax = axes.flat[i]
        t.score_type = ZScore
        model = dtc.dtc_to_model()
        score = t.judge(model)
        collect_z_offset.append(np.abs(float(score.raw)))

    for i,t in enumerate(random.tests):
        #ax = axes.flat[i]
        t.score_type = ZScore
        model = dtc.dtc_to_model()
        score = t.judge(model)
        collect_z_offset_random.append(np.abs(float(score.raw)))


    x1 = -1.01
    x2 = 1.0
    sigma = 1.0
    mu = 0
    x = np.arange(-sigma, sigma, 0.001) # range of x in spec
    x_all = np.arange(-sigma, sigma, 0.001)
    y_point = norm.pdf(mu+float(np.mean(collect_z_offset)),0,1)
    y2 = norm.pdf(x_all,0,1)

    y = norm.pdf(x,0,1)
    y2 = norm.pdf(x_all,0,1)
    x_point = mu+float(np.mean(collect_z_offset))

    x_point_random = mu+float(np.mean(collect_z_offset_random))
    y_point_random = norm.pdf(mu+float(np.mean(collect_z_offset_random)),0,1)
    best_random = [x_point_random,y_point_random]

    zplot(x_point,y_point,'all_tests',more=best_random)
    #break

    #ax.scatter(x_point,y_point,c='r',s=300,marker='o')
    #name = t.name.split('Test')[0]

    #ax.plot(x_all,y2)
    #ax.set_xlim([-1.0,1.0])
    #ax.set_title(str(name)+str(' ')+str(t.observation['mean'].units))
    #fig.suptitle(str('Average Z over All tests'), fontsize=16)
    #st.pyplot()

    #return np.mean(collect_z_offset)


def zplot(x_point,y_point,title,area=0.68, two_tailed=True, align_right=False, more=None):
    """Plots a z distribution with common annotations
    Example:
        zplot(area=0.95)
        zplot(area=0.80, two_tailed=False, align_right=True)
    Parameters:
        area (float): The area under the standard normal distribution curve.
        align (str): The area under the curve can be aligned to the center
            (default) or to the left.
    Returns:
        None: A plot of the normal distribution with annotations showing the
        area under the curve and the boundaries of the area.
    """
    # create plot object
    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots()
    # create normal distribution
    norm = scs.norm()
    # create data points to plot
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)

    ax.plot(x, y)
    ax.scatter(x_point,y_point,c='g',s=150,marker='o')

    if more is not None:
        ax.scatter(more[0],more[1],c='b',s=150,marker='o')


    # code to fill areas for two-tailed tests
    if two_tailed:
        left = norm.ppf(0.5 - area / 2)
        right = norm.ppf(0.5 + area / 2)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')

        ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                        where=(x > left) & (x < right))
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12,
                 rotation=90, va="bottom", ha="right")
        plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                 fontsize=12, rotation=90, va="bottom", ha="left")
    # for one-tailed tests
    else:
        # align the area to the right
        if align_right:
            left = norm.ppf(1-area)
            ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x > left)
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left),
                     fontsize=12, rotation=90, va="bottom", ha="right")
        # align the area to the left
        else:
            right = norm.ppf(area)
            ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x < right)
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                     fontsize=12, rotation=90, va="bottom", ha="left")

    # annotate the shaded area
    plt.text(0, 0.1, "shaded area = {0:.3f}".format(area), fontsize=12,
             ha='center')
    # axis labels
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.title(title)
    plt.show()



def inject_model_soma(dtc,figname=None,solve_for_current=None,fixed=False):
    from neuronunit.tests.target_spike_current import SpikeCountSearch

    # get rheobase injection value
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    if type(solve_for_current) is not type(None):
        observation_range={}
        model = dtc.dtc_to_model()
        temp = copy.copy(model.attrs)

        if not fixed:
            observation_range['value'] = dtc.spk_count
            scs = SpikeCountSearch(observation_range)
            target_current = scs.generate_prediction(model)
            if type(target_current) is not type(None):
                solve_for_current = target_current['value']
            dtc.solve_for_current = solve_for_current
            ALLEN_DELAY = 1000.0*qt.ms
            ALLEN_DURATION = 2000.0*qt.ms
        #print(solve_for_current,model.attrs,'changing?',observation_range)
        uc = {'amplitude':solve_for_current,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
        model = dtc.dtc_to_model()
        model._backend.attrs = temp
        model.inject_square_current(**uc)
        if hasattr(dtc,'spikes'):
            dtc.spikes = model._backend.spikes
        #print(len(model._backend.spikes),'not enough spikes')
        vm15 = model.get_membrane_potential()
        dtc.vm_soma = copy.copy(vm15)

        del model
        return vm15,uc,None,dtc
    #print('falls through \n\n\n')


    if dtc.rheobase is None:
        rt = RheobaseTest(observation={'mean':0*pq.pA,'std':0*pq.pA})
        dtc.rheobase = rt.generate_prediction(dtc.dtc_to_model())
        if dtc.rheobase is None:
            return None,None,None,None,dtc
    model = dtc.dtc_to_model()
    if type(dtc.rheobase) is type(dict()):
        if dtc.rheobase['value'] is None:
            return None,None,None,None,dtc
        else:
            rheobase = dtc.rheobase['value']
    else:
        rheobase = dtc.rheobase
    model = dtc.dtc_to_model()
    ALLEN_DELAY = 1000.0*qt.s
    ALLEN_DURATION = 2000.0*qt.s
    #print('before, crash out b ',rheobase)
    uc = {'amplitude':rheobase,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
    model._backend.inject_square_current(**uc)
    dtc.vmrh = None
    dtc.vmrh = model.get_membrane_potential()
    del model



    model = dtc.dtc_to_model()
    ########
    # a big thing to note
    ##
    # rheobase = 300 * pq.pA
    ##
    # a big thing to note
    #     #####
    ##
    # uc = {'amplitude':3.0*rheobase,'duration':DURATION,'delay':DELAY}
    # model.inject_square_current(uc)
    # vm30 = model.get_membrane_potential()
    # dtc.vm30 = copy.copy(vm30)
    # del model
    ##
    model = dtc.dtc_to_model()
    if hasattr(dtc,'current_spike_number_search'):
        from neuronunit.tests import SpikeCountSearch
        observation_spike_count={}
        observation_spike_count['value'] = dtc.current_spike_number_search
        scs = SpikeCountSearch(observation_spike_count)
        assert model is not None
        target_current = scs.generate_prediction(model)

        uc = {'amplitude':target_current,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
        model.inject_square_current(uc)
        vm15 = model.get_membrane_potential()
        dtc.vm_soma = copy.copy(vm15)

        del model
        model = dtc.dtc_to_model()
        uc = {'amplitude':0*pq.pA,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
        params = {'amplitude':rheobase,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
        model.inject_square_current(uc)
        vr = model.get_membrane_potential()
        dtc.vmr = np.mean(vr)
        del model
        return vm30,vm15,params,None,dtc

    else:
            #find_target_current
        #print('before, crash out c ',rheobase)

        uc = {'amplitude':1.5*rheobase,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
        model._backend.inject_square_current(**uc)
        vm15 = model.get_membrane_potential()
        dtc.vm_soma = copy.copy(vm15)
        del model
        model = dtc.dtc_to_model()
        #print('before, crash out d ',rheobase)

        uc = {'amplitude':3.0*rheobase,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
        #model.inject_square_current(uc)
        model._backend.inject_square_current(**uc)

        vm30 = model.get_membrane_potential()
        dtc.vm30 = copy.copy(vm30)
        #print('still gets here mayhem \n\n',type(solve_for_current))

        #print('still gets here mayhem',dtc.vm30,type(solve_for_current))
        #print('before, crash out e ',rheobase)

        del model
        model = dtc.dtc_to_model()
        uc = {'amplitude':00*pq.pA,'duration':DURATION,'delay':DELAY}
        params = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
        #model.inject_square_current(uc)
        model._backend.inject_square_current(**uc)

        vr = model.get_membrane_potential()
        dtc.vmr = np.mean(vr)
        del model
        return vm30,vm15,params,None,dtc
from elephant.spike_train_generation import threshold_detection


def display_fitting_data():
    cells = pickle.load(open("processed_multicellular_constraints.p","rb"))

    purk = TSD(cells['Cerebellum Purkinje cell'])#.tests
    purk_vr = purk["RestingPotentialTest"].observation['mean']

    ncl5 = TSD(cells["Neocortex pyramidal cell layer 5-6"])
    ncl5.name = str("Neocortex pyramidal cell layer 5-6")
    ncl5_vr = ncl5["RestingPotentialTest"].observation['mean']

    ca1 = TSD(cells['Hippocampus CA1 pyramidal cell'])
    ca1_vr = ca1["RestingPotentialTest"].observation['mean']


    olf = TSD(pickle.load(open("olf_tests.p","rb")))
    olf.use_rheobase_score = False
    cells.pop('Olfactory bulb (main) mitral cell',None)
    cells['Olfactory bulb (main) mitral cell'] = olf

    #constraints= cells

    list_of_dicts = []
    for k,v in cells.items():
        observations = {}
        for k1 in ca1.keys():
            vsd = TSD(v)
            if k1 in vsd.keys():
                vsd[k1].observation['mean']
                observations[k1] = float(vsd[k1].observation['mean'])##,2)

                observations[k1] = np.round(vsd[k1].observation['mean'],2)
                observations['name'] = k
        list_of_dicts.append(observations)
    df = pd.DataFrame(list_of_dicts)
    #df
    #df.round(decimals=3)

    df = df.set_index('name').T

    return df


def efel_evaluation(dtc,thirty=False):
    if hasattr(dtc,'solve_for_current'):
        current = dtc.solve_for_current#['value']
    else:
        if type(dtc.rheobase) is type(dict()):
            rheobase = dtc.rheobase['value']
        else:
            rheobase = dtc.rheobase
        if not thirty:
            current = 1.5*float(rheobase)
        else:
            current = 3.0*float(rheobase)

    if not thirty:
        vm_used = dtc.vm_soma
    else:
        vm_used = dtc.vm30

    #print(np.min(vm_used),np.max(vm_used))
    #fig = apl.figure()
    #fig.plot([float(t)*1000.0 for t in vm_used.times],[float(v) for v in vm_used.magnitude],label=str(dtc.attrs), width=100, height=20)
    #fig.show()

    #print(np.max(vm_used),np.mean(vm_used),np.std(vm_used))

    try:
        efel.reset()
    except:
        pass


    efel.setThreshold(0)

    #return list(thresh_cross)

    trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
              'V': [float(v) for v in vm_used.magnitude],
              'stimulus_current': [current]}
    ALLEN_DURATION = 2000*qt.ms
    ALLEN_DELAY = 1000*qt.ms

    trace3['stim_end'] = [ float(ALLEN_DELAY)+float(ALLEN_DURATION) ]
    trace3['stim_start'] = [ float(ALLEN_DELAY)]

    #simple_yes_list = [,'mean_frequency','adaptation_index2',,'median_isi','AHP_depth_abs','sag_ratio2','ohmic_input_resistance','sag_ratio2','peak_voltage','voltage_base','Spikecount','ohmic_input_resistance_vb_ssse']
    efel_list = list(efel.getFeatureNames())
    #reduced_list = [ i for i in efel_list if i in simple_yes_list ]
    #print('gets to a')
    #print(np.min(vm_used.magnitude)<0 and np.max(vm_used.magnitude)>0)
    if np.min(vm_used.magnitude)<0:
        if not np.max(vm_used.magnitude)>0:
            vm_used_mag = [v+np.mean([0,float(np.max(v))])*pq.mV for v in vm_used]
            if not np.max(vm_used_mag)>0:
                dtc.efel_15 = None
                return dtc

            trace3['V'] = vm_used_mag
            #print('gets to b')
            #print(np.min(vm_used.magnitude)<0 and np.max(vm_used.magnitude)>0)

            # and np.max(vm_used.magnitude)>0:
        else:
            pass

        #try:
        specific_filter_list = [
                    'burst_ISI_indices',
                    'burst_mean_freq',
                    'burst_number',
                    'single_burst_ratio',
                    'ISI_log_slope',
                    'mean_frequency',
                    'adaptation_index2',
                    'first_isi',
                    'ISI_CV',
                    'median_isi',
                    'Spikecount',
                    'all_ISI_values',
                    'ISI_values',
                    'time_to_first_spike',
                    'time_to_last_spike',
                    'time_to_second_spike',
                    'Spikecount']
        '''
        'voltage',
        'AHP_depth_abs'
        'AHP_depth',
        'AHP_depth_abs',
        'AHP_depth_last',
        'sag_ratio2',
        'bus',
        'sag_ratio2',
        'peak_voltage',
        'voltage_base',
        '''
        results = efel.getMeanFeatureValues([trace3],specific_filter_list)#, parallel_map=pool.map)
        if "MAT" not in dtc.backend:
            thresh_cross = threshold_detection(vm_used,0*qt.mV)
            for index,tc in enumerate(thresh_cross):
                results[0]['spike_'+str(index)]=float(tc)
        else:
            #print('gets here')
            if hasattr(dtc,'spikes'):
                dtc.spikes = model._backend.spikes
                for index,tc in enumerate(dtc.spikes):
                    results[0]['spike_'+str(index)]=float(tc)

            #print(tc,tc.units)

        nans = {k:v for k,v in results[0].items() if type(v) is type(None)}
        if thirty:
            dtc.efel_30 = None
            dtc.efel_30 = results

        else:


            dtc.efel_15 = None
            dtc.efel_15 = results
        efel.reset()
    #print(dtc.efel_15)
    return dtc
    '''
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
    trace15['T'] = [float(t) for t in model.vm_soma.times.rescale('ms')]
    trace15['V'] = [float(v) for v in model.vm_soma.magnitude]
    trace15['stim_start'] = [ trace15['T'][0] ]
    trace15['stimulus_current'] = [ model.druckmann2013_standard_current ]
    trace15['stim_end'] = [ trace15['T'][-1] ]
    traces15 = [trace15]# Now we pass 'traces' to the efel and ask it to calculate the feature# values

    ##
    # Compute
    # EFEL features (HBP)
    ##
    efel.reset()

    if len(threshold_detection(model.vm_soma, threshold=0)):
        threshold = float(np.max(model.vm_soma.magnitude)-0.5*np.abs(np.std(model.vm_soma.magnitude)))


    #efel_15 = efel.getMeanFeatureValues(traces15,list(efel.getFeatureNames()))#
    else:
        threshold = float(np.max(model.vm_soma.magnitude)-0.2*np.abs(np.std(model.vm_soma.magnitude)))
    efel.setThreshold(threshold)
    if np.min(model.vm_soma.magnitude)<0:
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
    return dtc
    '''

def dtc_to_predictions(dtc):
    dtc.preds = {}
    for t in dtc.tests:
        preds = t.generate_prediction(dtc.dtc_to_model())
        dtc.preds[t.name] = preds
    return dtc

def evaluate_allen(dtc):
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    fitness = [ 1.0 for i in range(0,len(dtc.ascores)) ]
    if dtc.ascores[str(t)] is None:
        fitness[int_] = 1.0
    else:
        fitness[int_] = dtc.ascores[str(t)]
    return tuple(fitness,)

def evaluate(dtc,allen=None):
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    if not hasattr(dtc,str('SA')):
        return []
    else:
        if allen is not None:
            not_allen = [ t for t in dtc.tests if not hasattr(t,'allen') ]
            fitness = []
            dtc.tests = not_allen
            fitness = [v for v in dtc.SA.values]
            return augment_with_three_step(dtc,fitness)
        if allen is None:
            fitness = (v for v in dtc.SA.values)
            return fitness
def check_bin_vm30(target,opt):
    plt.plot(target.vm30.times,target.vm30.magnitude)
    plt.plot(opt.vm30.times,opt.vm30.magnitude)
    signal = target.vm30
    plt.xlabel(qt.s)
    plt.ylabel(signal.dimensionality)

    plt.show()
def check_bin_vm15_uc(target,opt,uc):
    plt.plot(target.vm_soma.times,target.vm_soma.magnitude,label='Allen Experiment')
    plt.plot(opt.vm_soma.times,opt.vm_soma.magnitude,label='Optimized Model')
    signal = target.vm_soma
    plt.xlabel(qt.s)
    plt.ylabel(signal.dimensionality)
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot([t for t in uc], uc)
    plt.legend()
    plt.show()
def check_bin_vm_soma(target,opt):
    plt.plot(target.vm_soma.times,target.vm_soma.magnitude,label='Allen Experiment')
    plt.plot(opt.vm_soma.times,opt.vm_soma.magnitude,label='Optimized Model')
    signal = target.vm_soma
    plt.xlabel(qt.s)
    plt.ylabel(signal.dimensionality)
    plt.legend()
    plt.show()

    #[ uc['delay']*0, uc['duration'] * uc['amp'],


from sciunit.scores import ZScore, RatioScore

def augment_with_three_step(dtc,fitness):
    temp_tests = copy.copy(dtc.tests)
    allen = [ t for t in temp_tests if hasattr(t,'allen') ]
    dtc = three_step_protocol(dtc)
    features = dtc.preds
    dtc.tests = temp_tests
    for t in allen:
        t.score_type = ZScore
    for t,(k,v) in zip(allen,features.items()):
        if t.name == k:
            t.set_prediction(v)
    for t in allen:
        if type(t.prediction['mean']) is type(dict()):
            x = t.prediction['mean']['mean']
        else:
            x = t.prediction['mean']
        try:
            result = np.abs(np.log(np.abs(x-t.observation['mean'])/t.observation['std']))
        except:
            try:
                model = dtc.dtc_to_model()
                result = t.judge(model)
            except:
                result = 1000.0
        fitness.append(result)
    for i,f in enumerate(fitness):
        if np.isnan(f) or np.isinf(f):
            fitness[i] = 1000.0
    return (f for f in fitness)


def evaluate_old(dtc):
    """
    optimize against means of observations
    don't use SciUnit lognorm score
    NU test results are still computed
    but they are not optimized against.
    """
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    if not hasattr(dtc,str('SA')):
        return []
    else:
        temp_tests = copy.copy(dtc.tests)
        not_allen = [ t for t in dtc.tests if not hasattr(t,'allen') ]
        fitness = []
        dtc.tests = not_allen
        fitness = [v for v in dtc.SA.values]
        return augment_with_three_step(dtc,fitness)




def evaluate_260(observations_260,dtc):
    """
    optimize against means of observations
    don't use SciUnit lognorm score
    NU test results are still computed
    but they are not optimized against.
    """
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    if not hasattr(dtc,str('SA')):
        return []
    else:
        dtc = three_step_protocol(dtc)
        dtc.get_agreement()
        prefix = dtc.agreement
        fitness = []
        a_lot_of_predictions = dtc.everything
        for k,v in a_lot_of_predictions.items():
            temp = (observations_260[k] - v)
            temp = np.abs(float(temp.simplified))
            fitness.append(temp)
        return fitness

def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list

from sciunit import scores

#Decorator is dumb makes function harder to
#use normaly and generally unflexible.
@dask.delayed
def transform_delayed(xargs):
    (ind,td,backend) = xargs
    dtc = DataTC()
    dtc.attrs = {}
    for i,j in enumerate(ind):
        dtc.attrs[str(td[i])] = j
    dtc.evaluated = False
    return dtc

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
    both = [ (d,p) for d,p in zip(dtcpop,pop) if type(d.rheobase) is not type(None) ]
    both = [ (d,p) for d,p in both if type(p.rheobase) is not type(None) ]
    pop = [i[1] for i in both]
    dtcpop = [i[0] for i in both]

    assert len(pop) == len(dtcpop), print('fatal')
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
    if 'mean' in thing.keys():
        return 'mean'
    if 'value' in thing.keys():
        return 'value'

"""
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
"""

def score_attr(dtcpop,pop):
    for i,d in enumerate(dtcpop):
        if not hasattr(pop[i],'dtc'):
            pop[i].dtc = None
        d.get_ss()
        pop[i].dtc = copy.copy(d)
    return dtcpop,pop

from dask import compute, delayed
def get_dm(dtcpop,pop=None):
    if PARALLEL_CONFIDENT:
        NPART = min(npartitions,len(dtcpop))
        dtcbag = [ delayed(nuunit_dm_evaluation(d)) for d in dtcpop ]
        dtcpop = compute(*dtcbag)

    else:
        dtcpop = list(map(nuunit_dm_evaluation,dtcpop))
    if type(pop) is not type(None):
        dtcpop,pop = score_attr(dtcpop,pop)
    return dtcpop,pop


from sciunit.scores.collections import ScoreArray
import numpy as np
#dir(stats['best_random_model'].SA['RheobaseTest'])
#from sciunit.scores.collections import ScoreArray

#from sciunit.scores.collections_m2m import  ScoreMatrixM2M,  ScoreArrayM2M#(pd.DataFrame, SciUnit, TestWeighted)

class OptMan():
    def __init__(self, \
                tests, \
                td=None, \
                backend = None,
                hc = None,\
                boundary_dict = None, \
                error_length=None,\
                protocol=None,\
                simulated_obs=None,\
                verbosity=None,\
                PARALLEL_CONFIDENT=True,\
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
        self.NGEN = None
        self.MU = None

        if self.backend is "IZHI" or self.backend is "ADEXP" or self.backend is "NEURONHH":
            self.PARALLEL_CONFIDENT = False
            self.MEMORY_FRIENDLY = True

        else:
            self.MEMORY_FRIENDLY = False

            self.PARALLEL_CONFIDENT = True
        if verbosity is None:
            self.verbose = 0
        else:
            self.verbose = verbosity
        if type(tsr) is not None:
            self.tsr = tsr

    def random_sample(self,dtc,search_size):
        import shelve

        d = shelve.open('random_sample_models_cache')  # open -- file may get suffix added by low-level
        flag = False
        query_key = None
        if hasattr(self,'kwargs'):
            kwargs = self.kwargs

            query_key = str(kwargs['free_parameters']) +\
            str(kwargs['backend']) +\
            str(kwargs['protocol']) +\
            str(kwargs['hold_constant'])
            flag = query_key in d
            flag = False
            if flag:
                stats = d[query_key]
                d.close()
                del d
                return stats


        if not flag:
            d.close()
            del d
            pop,dtcpop = self.boot_new_genes(search_size,dtc)
            dtcpop = [d for d in dtcpop if type(d.rheobase) is not type(None)]
            for d in dtcpop: d.tests = self.tests
            OM = dtcpop[0].dtc_to_opt_man()
            self.PARALLEL_CONFIDENT = False
            self.PARALLEL_CONFIDENT = True

            if self.PARALLEL_CONFIDENT and str('L5PC') not in  self.backend:
                dtcbag = db.from_sequence(dtcpop,NPART)
                dtcpop = list(dtcbag.map(self.format_test).compute())
                dtcbag = db.from_sequence(dtcpop,NPART)
                dtcpop = list(dtcbag.map(self.seval).compute())

            elif self.MEMORY_FRIENDLY:#
                passed = False
                lazy = []
                for d in dtcpop:
                    d = self.format_test(d)

                lazy = (dask.delayed(self.seval)(d) for d in dtcpop)
                dtcpop = list(dask.compute(*lazy))

            container = {}
            stats = {}
            dtcpop = [dtc for dtc in dtcpop if dtc.SA is not None ]
            for k,v in dtcpop[0].SA.items():
                container[k.name] = []
            cnt = 0

            sum_list = []
            for d in dtcpop:
                sum_list.append((np.sum([v for v in d.SA.values ]),cnt))
                cnt+=1
            sorted_sum_list = sorted(sum_list,key=lambda tup: tup[0])
            best = sorted_sum_list[0]
            stats['best_random_model'] = dtcpop[best[1]]
            mean_random=[]

            for t in stats['best_random_model'].tests:
                model = stats['best_random_model'].dtc_to_model()
                score = t.judge(model)
                #print(score.raw)
                mean_random.append(np.abs(float(score.raw)))
            SA = ScoreArray(stats['best_random_model'].tests,mean_random)
            stats['best_random_model_raw_score'] = SA
                #print(score.raw_score)
            stats['mean_best_random_model'] = np.mean(mean_random)

            for d in dtcpop:

                for k,v in d.SA.items():

                    if type(v) is not type(None):
                        try:
                            lns = np.abs(v.raw_score)
                            if lns==-np.inf or lns==np.inf:
                                lns = np.abs(v.raw_score)
                        except:
                            if type(v) is type(float()):
                                lns = v
                            else:
                                lns = np.abs(v.raw_score)
                        container[k.name].append(lns)
                for k,v in container.items():
                    try:
                        stats[k] = (np.mean(container[k]),np.std(container[k]),np.var(container[k]))
                    except:
                        stats[k] = (100.0,0.0,0.0)
            stats['best_random_sum_total'] = np.min(sum_list)

            stats['models'] = dtcpop
            temp = {k:v[0] for k,v in stats.items() if isinstance(v,Iterable)}
            frame = pd.DataFrame([temp])
            stats['frame'] = frame
            d = shelve.open('random_sample_models_cache')  # open -- file may get suffix added by low-level
            if query_key is None:
                query_key = str(dtc.backend)+str(search_size)+str(list(dtc.attrs.keys()))
            d[query_key] = stats
            return stats

    def get_dummy_tests(self):
        from neuronunit.optimization import get_neab
        tests = get_neab.process_all_cells()
        for t in tests.values():
            helper_tests = [value for value in t.tests ]
            break
        self.helper_tests = helper_tests

    def get_agreement(self,dtc):
        obs = {}
        pred = {}
        dtc.obs_preds = None
        dtc.obs_preds = {}
        if type(dtc.tests) is type(list()):
            temp = {t.name:t for t in dtc.tests}
        else:
            temp = dtc.tests
        if dtc.rheobase is not None:
            scores_d = {}
            for k in dtc.SA.keys():
                if hasattr(dtc.SA[k],'score'):
                    scores_d[str(k)] = dtc.SA[k].score
                else:
                    scores_d[str(k)] = dtc.SA[k]
                    scores_d["total"] = np.sum([ np.abs(v) for v in scores_d.values()])

            pre = len(temp)
            post = len({k:v for k,v in temp.items() if hasattr(v,'prediction')})
            if pre == post:

                similarity,lps,rps =  self.closeness(temp,temp)
                scores_ = {}
                for k,p,o in zip(list(similarity.keys()),lps,rps):
                    obs[str(k)] = o
                    pred[str(k)] = p
                dtc.agreement = pd.DataFrame([obs,pred,scores_d],index=['observations','predictions','scores'])
            else:
                print('sys log no prediction')
        #dtc.agreement = dtc.obs_preds

        return dtc


    def active_values(keyed,rheobase):#:,square = None):
        if isinstance(rheobase,type(dict())):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase
        return keyed

    @cython.boundscheck(False)
    @cython.wraparound(False)
    #@timer
    def make_static_threshold(self,dtc):

        model.attrs = model._backend.default_attrs
        ##
        #
        ##
        thresh = model.get_AP_thresholds()
        self.STATIC_THRESHOLD = tresh
        PASSIVE_DURATION = 500.0*pq.ms
        PASSIVE_DELAY = 200.0*pq.ms


    def format_test(self,dtc):
        model = dtc.dtc_to_model()

        ##
        # self.make_static_threshold(dtc)
        ##
        '''
        pre format the current injection dictionary based on pre computed
        rheobase values of current injection.
        This is much like the hooked method from the old get neab file.
        '''
        if type(dtc) is type(str()):
            print('error dtc is string')
        if not hasattr(dtc,'tests'):
            dtc.tests = copy.copy(self.tests)
        if isinstance(dtc.tests,type(dict())):
            for k,t in dtc.tests.items():
                assert 'std' in t.observation.keys()
            tests = [key for key in dtc.tests.values()]
            dtc.tests = switch_logic(tests)
        else:
            dtc.tests = switch_logic(dtc.tests)
        for v in dtc.tests:
            k = v.name
            #print(k,'optman')


            if not hasattr(v,'params'):
                v.params = {}
            if not 'injected_square_current' in v.params.keys():
                v.params['injected_square_current'] = {}
            if v.passive == False and v.active == True:
                keyed = v.params['injected_square_current']
                v.params = active_values(keyed,dtc.rheobase)
                v.params['injected_square_current']['delay'] = DELAY
                v.params['injected_square_current']['duration'] = DURATION
            if v.passive == True and v.active == False:

                v.params['injected_square_current']['amplitude'] =  -10*pq.pA
                v.params['injected_square_current']['delay'] = PASSIVE_DELAY
                v.params['injected_square_current']['duration'] = PASSIVE_DURATION

            if v.name in str('RestingPotentialTest'):
                v.params['injected_square_current']['delay'] = PASSIVE_DELAY
                v.params['injected_square_current']['duration'] = PASSIVE_DURATION
                v.params['injected_square_current']['amplitude'] = 0.0*pq.pA
            keyed = v.params['injected_square_current']
            try:
                v.params['t_max'] = keyed['delay']+keyed['duration'] + 200.0*pq.ms
            except:
                pass
            '''
            if str("InjectedCurrentAPThresholdTest") in v.name and not dtc.threshold:
                model = dtc.dtc_to_model()
                model.set_attrs(model._backend.default_attrs)
                model.attrs = model._backend.default_attrs
                #print(model.attrs)
                threshold = v.generate_prediction(model)
                dtc.threshold = threshold

            if str("APThresholdTest") in v.name:
                v.threshold = None
                v.threshold = dtc.threshold
            '''
                #print(v.threshold)

        return dtc




    def new_single_gene(self,dtc,td):
        from neuronunit.optimization.optimizations import SciUnitoptimization

        random.seed(datetime.now())
        try:
            DO = SciUnitoptimization(offspring_size = 1,
                error_criterion = [self.tests], boundary_dict = dtc.boundary_dict,
                                         backend = dtc.backend,simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)
        except:
            DO = SciUnitoptimization(offspring_size = 1,
                error_criterion = [dtc.preds], boundary_dict = dtc.boundary_dict,
                                         backend = dtc.backend, simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)

        DO.setnparams(nparams = len(dtc.attrs), boundary_dict = self.boundary_dict)
        DO.setup_deap()
        gene = DO.set_pop(boot_new_random=1)

        dtc_ = self.update_dtc_pop(gene)#,self.td)
        #dtc_ = pop2dtc(gene,dtc_)
        return gene[0], dtc_[0]

    def run_simple_grid(self,npoints=10,free_parameters=None):
        self.exhaustive = True
        from neuronunit.optimization.exhaustive_search import sample_points, add_constant, chunks
        ranges = self.boundary_dict
        subset = OrderedDict()
        if type(free_parameters) is type(None):
            free_parameters = list(ranges.keys())
        for k,v in ranges.items():
            if k in free_parameters:
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
        if self.PARALLEL_CONFIDENT:
            dtcbag = db.from_sequence(dtcpop,NPART)
            dtcpop = list(dtcbag.map(nuunit_allen_evaluation).compute())
            #dtcbag = [ delayed(nuunit_allen_evaluation(d)) for d in dtcpop ]
            #dtcpop = compute(*dtcbag)

            #dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
            #dtcpop = list(dtcbag.map(nuunit_allen_evaluation).compute())

        elif self.MEMORY_FRIENDLY:
            #self.MEMORY_FRIENDLY = True
            dtcbag = [ delayed(nuunit_allen_evaluation(d)) for d in dtcpop ]
            dtcpop = compute(*dtcbag)

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

        if self.PARALLEL_CONFIDENT:
            # switch to delayed syntax
            #dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
            #dtcpop = list(dtcbag.map(nuunit_allen_evaluation).compute())

            dtcbag = [ delayed(nuunit_allen_evaluation(d)) for d in dtcpop ]
            dtcpop = compute(*dtcbag)


        else:
            dtcpop = list(map(nuunit_allen_evaluation,dtcpop))

        return pop, dtcpop

    def closeness(self,left,right):
        closeness_ = {}
        lps = []
        rps = []
        for k in left.keys():
            key = which_key(left[k].prediction)
            lp = left[k].prediction[key]
            key = which_key(right[k].observation)
            rp = right[k].observation[key]
            if lp is not None:
                lp = lp.rescale(rp.units)
                closeness_[k] = np.abs(lp-rp)

            else:
                closeness_[k] = None
            rps.append(rp)
            lps.append(lp)

        return closeness_,lps,rps

    def make_sim_data_tests(self,backend,free_parameters=None,test_key=None,protocol=None):
        #print(test_key)
        ###
        # new code
        ###
        with open('processed_multicellular_constraints.p','rb') as f:
            test_frame = pickle.load(f)
        stds = {}
        for k,v in TSD(test_frame['Neocortex pyramidal cell layer 5-6']).items():
            temp = TSD(test_frame['Neocortex pyramidal cell layer 5-6'])[k]
            stds[k] = temp.observation['std']

        OMObjects = []
        '''
        if "FITest" in test_key:
            from neuronunit.tests.dynamics import FITest

            test_frame['Neocortex pyramidal cell layer 5-6'].tests.append(FITest(observation={'mean':10*pq.Hz/pq.pA,'std':10*pq.Hz/pq.pA}))
        '''
        cloned_tests = copy.copy(test_frame['Neocortex pyramidal cell layer 5-6'])
        ### new code
        ####

        OM = jrt(cloned_tests,backend,protocol=protocol)
        x= {k:v for k,v in OM.tests.items() if 'mean' in v.observation.keys() or 'value' in v.observation.keys()}
        cloned_tests = copy.copy(OM.tests)
        #print(test_key)
        if test_key is not None:
            if len(test_key)==1:
                #print(test_key)
                OM.tests = TSD({test_key:cloned_tests[test_key]})
            else:
                OM.tests = TSD({tk:cloned_tests[tk] for tk in test_key})

        else:
            OM.tests = TSD(cloned_tests)
        #import pdb; pdb.set_trace()
        rt_out = OM.simulate_data(OM.tests,OM.backend,free_parameters=free_parameters)
        #import pdb; pdb.set_trace()

        target = rt_out[0]
        penultimate_tests = TSD(test_frame['Neocortex pyramidal cell layer 5-6'])
        '''
        if "FITest" in test_key:
            penultimate_tests["FITest"] = FITest(observation={'mean':10*pq.Hz/pq.pA,'std':10*pq.Hz/pq.pA})
            #import pdb
            #pdb.set_trace()
            #test_frame['Neocortex pyramidal cell layer 5-6'].tests.append(FITest(observation={'mean':10*pq.Hz/pq.pA,'std':10*pq.Hz/pq.pA}))
        '''
        for k,v in OM.tests.items():
            if k in rt_out[1].keys():
                v = rt_out[1][k].observation
                if k not in stds.keys():
                    v['std'] = v['value']
                else:
                    v['std'] = stds[k]
        simulated_data_tests = TSD(OM.tests)

        target.tests = simulated_data_tests
        target = self.format_test(target)
        for t in simulated_data_tests.values():


            if str('value') in t.observation.keys() and str('mean') not in  t.observation.keys():
                t.observation['mean'] =  t.observation['value']


            #import sciunit.capabilities as scap

            t.score_type = scores.ZScore
            model = target.dtc_to_model()
            score = t.judge(model)
            pred = t.generate_prediction(model)
            #print(score.score)

            assert float(score.score)==0.0

        # # Show what the randomly generated target waveform the optimizer needs to find actually looks like
        # # first lets just optimize over all objective functions all the time.
        # # Commence optimization of models on simulated data sets
        #simulated_data_tests.update(rt_out[2])

        return simulated_data_tests, OM, target


    def simulate_data(self,
                        tests,
                        backend,
                        free_parameters=None):
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
        #print('does not get here')
        #import pdb; pdb.set_trace()

        out_tests = []
        ranges = MODEL_PARAMS[backend]

        if self.protocol['allen']:
            new_tests = False
            cnt = 0

            while new_tests is False:

                self.protocol = {'allen':False,'elephant':True}

                dsolution,rp,_,_ = process_rparam(backend,
                                                free_parameters=free_parameters)
                (new_tests,dtc) = self.make_simulated_observations(tests,
                                                                    backend,rp,
                                                                    dsolution=dsolution)
                #if cnt % 5 ==0:
                #    #print(dsolution,rp,new_tests,'stuck')
                if new_tests is False:
                    continue

                new_tests = {k:v for k,v in new_tests.items() if v.observation[which_key(v.observation)] is not None}
                if type(new_tests) is not type(bool()):
                    for t in new_tests.values():
                        key = which_key(t.observation)
                        if t.observation[key] is None:
                            new_tests = False
                            break
                    if type(new_tests) is type(bool()):
                        continue
                    if 'RheobaseTest' not in new_tests.keys():
                        new_tests = False
                        continue
                    dsolution.rheobase = new_tests['RheobaseTest'].observation
                    if dsolution.rheobase['value'] is not None:
                        if float(dsolution.rheobase['value'])==0:
                            new_tests = False
                            continue
                    else: # Rheobase is None
                        new_tests = False
                        continue
                    temp_vm = inject_and_not_plot_model(dsolution,known_rh=dsolution.rheobase)
                    if np.isnan(np.min(temp_vm)):
                        new_tests = False
                        continue

                    try:
                        target = dsolution
                        target = three_step_protocol(target)
                        #self.protocol = {'allen':True,'elephant':False}

                    except:
                        continue
                cnt+=1




            from neuronunit.tests.base import VmTest
            allen_tests = []
            cleaned0 = { key:value['mean'] for key,value in target.preds.items() if hasattr(value,'keys')}
            cleaned1 = { key:value for key,value in target.preds.items() if not hasattr(value,'keys')}
            cleaned0.update(cleaned1)
            for key,value in cleaned0.items():
                temp = {}
                #for key,value in target.preds.items():
                generic = VmTest(observation=temp,name = key)
                generic.observation['mean'] = value
                generic.observation['std'] = value
                generic.allen = None
                generic.allen = True

                allen_tests.append(generic)

            target.allen_tests = allen_tests
            more = TSD(allen_tests)
            new_tests.update(more)
            print('Random simulated data tests made',new_tests)

            return target,new_tests, more


        elif self.protocol['elephant']:
            new_tests = False
            cnt = 0

            while new_tests is False:
                cnt+=1
                if cnt == 15:
                    print("search space has \
                     too many unstable modeles \
                      try refining model parameter edges")
                    stats = self.random_sample(dsolution,55)
                    dtcpop = stats['models']
                    # find element of dtcpop where all tests return a sensible score
                    # no reason to believe dtcpop[0] has succeeded in this way.
                    rp = dtcpop[0].attrs.values

                    (new_tests,dtc) = self.make_simulated_observations(tests,dtcpop[0].backend,rp,dsolution=dsolution)
                    break

                dsolution,rp,_,_ = process_rparam(backend,free_parameters=free_parameters)

                dsolution = dtc_to_rheo(dsolution)
                (new_tests,dtc) = self.make_simulated_observations(tests,backend,rp,dsolution=dsolution)
                if cnt % 5 ==0:
                    print(dsolution,rp,new_tests,'stuck')
                if new_tests is False:
                    continue
                new_tests = {k:v for k,v in new_tests.items() if v.observation[which_key(v.observation)] is not None}
                if type(new_tests) is not type(bool()):
                    for t in new_tests.values():
                        key = which_key(t.observation)
                        if t.observation[key] is None:
                            new_tests = False
                            break
                    if type(new_tests) is type(bool()):
                        continue
                    if 'RheobaseTest' not in new_tests.keys():
                        new_tests = False
                        continue
                    dsolution.rheobase = new_tests['RheobaseTest'].observation
                    if dsolution.rheobase['value'] is not None:
                        if float(dsolution.rheobase['value'])==0:
                            new_tests = False
                            continue
                    else: # Rheobase is None
                        new_tests = False
                        continue
                    temp_vm = inject_and_not_plot_model(dsolution,known_rh=dsolution.rheobase)
                    if np.isnan(np.min(temp_vm)):
                        new_tests = False
                        continue
            print('Random simulated data tests made')
            target = dsolution
            return target,new_tests

    def grid_search(self,explore_ranges,test_frame,backend=None):
        '''
        Hopefuly this method can depreciate the whole file optimization_management/exhaustive_search.py
        Well actually not quiete. This method does more than that. It iterates over multiple NeuroElectro datum entities.
        A more generalizable method would just act on one NeuroElectro datum entities.
        '''
        store_results = {}
        npoints = 8
        grid = ParameterGrid(explore_ranges)

        size = len(grid)
        '''
        temp = []
        if size > npoints:
            sparsify = np.linspace(0,len(grid)-1,npoints)
            for i in sparsify:
                temp.append(grid[int(i)])
            grid = temp
        '''
        dtcpop = []
        for local_attrs in grid:
            store_results[str(local_attrs.values())] = {}
            dtc = DataTC()
            #dtc.tests = use_test
            dtc.attrs = local_attrs

            dtc.backend = backend433

            dtcpop.append(dtc)


        for key, use_test in test_frame.items():
            for dtc in dtcpop:
                dtc.tests = use_test
            '''
            if 'IZHI' in dtc.backend  or 'HH' in dtc.backend:#Backend:
                serial_faster = True # because of numba
                #dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
                #dtcpop = list(map(dtc_to_rheo,dtcpop))
                #if __name__ == '__main__':
                dtcbag = [ delayed(dtc_to_rheo(d)) for d in dtcpop ]
                dtcpop = compute(*dtcbag)

            else:
            '''
            serial_faster = False
            dtcpop = list(map(dtc_to_rheo,dtcpop))
            """
            The dask bag mapping works.
            It is faster, its just not the most memory
            friendly. It will exhaust RAM on HPC
            For MU>=150, NGEN>=150
            joblib has similar syntax
            and would probably work under same
            parallel circumstances without memory exhaustion
            """
            #dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
            #dtcpop = list(dtcbag.map(self.format_test))

            dtcbag = [ delayed(self.format_test(d)) for d in dtcpop ]
            dtcpop = compute(*dtcbag)
            dtcpop = [ dtc for dtc in dtcpop if type(dtc.rheobase) is not type(None) ]
            #dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
            #dtcpop = list(dtcbag.map(self.elephant_evaluation))
            dtcbag = [ delayed(self.elephant_evaluation(d)) for d in dtcpop ]
            dtcpop = compute(*dtcbag)
            #for d in dtcpop:
            #    print(d.SA)

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

    #@timer
    def pred_evaluation(self,dtc):
        # Inputs single data transport container modules, and neuroelectro observations that
        # inform test error error_criterion
        # Outputs Neuron Unit evaluation scores over error criterion
        dtc = copy.copy(dtc)
        # TODO
        # phase out model path:
        # via very reduced model
        #if hasattr(dtc,'model_path'):
            #dtc.model_path = path_params['model_path']

        dtc.preds = None
        dtc.preds = {}
        dtc = dtc_to_rheo(dtc)

        dtc = self.format_test(dtc)
        tests = dtc.tests
        if type(tests) is not type(TSD()) or type(tests) is type(list()):
            tests = {t.name:t for t in tests}
        for k,t in tests.items():
            #if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
            #t.params = dtc.protocols[k]
            test_and_models = (t, dtc)
            pred = pred_only(test_and_models)
            dtc.preds[str(t.name)] = pred
        return dtc

    #@timer
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
                dtcbag = [ delayed(OM.elephant_evaluation(d)) for d in dtcpop ]
                dtcpop = compute(*dtcbag)

                #bagged = db.from_sequence(dtcpop,npartitions=npartitions)
                #dtcpop = list(bagged.map(OM.elephant_evaluation))

            except:
                dtcpop = list(map(OM.elephant_evaluation,dtcpop))
            for i,j in enumerate(dtcpop):
                df.iloc[l][i] = np.sum(list(j.SA.values))/len(list(j.SA.values))

        return df

    def pred_std(self,pred,t):
        take_anything = list(pred.values())[0]
        if take_anything is None or type(take_anything) is type(int()):
            take_anything = list(pred.values())[1]

        return pred


    def preprocess(self,dtc):
        tests = dtc.tests
        if isinstance(tests,type(dict())):
            tests = list(tests.values())
        if isinstance(dtc.rheobase,type(None)) or type(dtc.rheobase) is type(None):
            dtc = allocate_worst(tests, dtc)
        else:
            for t in tests:
                k = str(t.name)
                try:
                    assert hasattr(self,'use_rheobase_score')
                except:
                    print('warning please add whether or not model should be scored on rheobase to protocol')
                    self.use_rheobase_score = True
                if self.use_rheobase_score == False and "RheobaseTest" in str(k):
                    continue
                #t.params = dtc.protocols[k]
                if not 'std' in t.observation.keys():
                    t.observation['std'] = copy.copy(t.observation['mean'])
                if float(t.observation['std']) == 0.0:
                    t.observation['std'] = copy.copy(t.observation['mean'])
            for i,t in enumerate(dtc.tests):
                k = str(t.name)
                if k == "RheobaseTestP":
                    if "RheobaseTestP" not in self.tests:
                        self.tests[k] = dtc.tests[i]
                assert dtc.tests[i].observation['mean'] == self.tests[k].observation['mean']

        return tests
    def elephant_evaluation(self,dtc):
        # Inputs single data transport container modules, and neuroelectro observations that
        # inform test error error_criterion
        # Outputs Neuron Unit evaluation scores over error criterion

        model = dtc.dtc_to_model()
        if not hasattr(dtc,'scores') or dtc.scores is None:
            dtc.scores = None
            dtc.scores = {}
            if hasattr(dtc,'SA'):
                pass
        dtc.tests = self.preprocess(dtc)
        for i,_ in enumerate(dtc.tests):
            assert dtc.tests[i].observation['mean'] == dtc.tests[i].observation['mean']

        scores_ = []
        suite = TestSuite(dtc.tests)
        for t in suite:
            if 'RheobaseTest' in t.name: t.score_type = sciunit.scores.ZScore
            if 'RheobaseTestP' in t.name: t.score_type = sciunit.scores.ZScore
            if 'mean' not in t.observation.keys():
                t.observation['mean'] = t.observation['value']
            try:
                score_gene = t.judge(model)
            except:
                score_gene = None
                lns = 100.0

            if not isinstance(type(score_gene),type(None)):
                if not isinstance(type(score_gene),sciunit.scores.InsufficientDataScore):
                    try:
                        lns = np.abs(score_gene.log_norm_score)
                    except:
                        lns = 100.0
                else:
                    lns = np.abs(float(score_gene.raw))
            scores_.append(lns)

        for i,s in enumerate(scores_):
            if s==np.inf:
                try:
                    scores_[i] = np.abs(float(score_gene.raw))
                except:
                    scores_[i] = 100.0
        dtc.SA = ScoreArray(dtc.tests, scores_)

        obs = {}
        pred = {}
        temp = {t.name:t for t in dtc.tests}

        if dtc.rheobase is not None:
            scores_d = {}
            for k in dtc.SA.keys():
                if hasattr(dtc.SA[k],'score'):
                    scores_d[k] = dtc.SA[k].score
                else:
                    scores_d[k] = dtc.SA[k]
                    scores_d["total"] = np.sum([ np.abs(v) for v in scores_d.values()])

            pre = len(temp)
            post = len({k:v for k,v in temp.items() if hasattr(v,'prediction')})
            if pre == post:
                similarity,lps,rps =  self.closeness(temp,temp)
                scores_ = {}
                for k,p,o in zip(list(similarity.keys()),lps,rps):
                    obs[k] = o
                    pred[k] = p
                dtc.obs_preds = pd.DataFrame([obs,pred,scores_d],index=['observations','predictions','scores'])
            else:
                print('sys log no prediction')
        assert dtc.SA is not None
        return dtc

    @dask.delayed
    def elephant_evaluation_delayed(self,dtc):
        # Inputs single data transport container modules, and neuroelectro observations that
        # inform test error error_criterion
        # Outputs Neuron Unit evaluation scores over error criterion

        model = dtc.dtc_to_model()
        if not hasattr(dtc,'scores') or dtc.scores is None:
            dtc.scores = None
            dtc.scores = {}
            if hasattr(dtc,'SA'):
                pass
        dtc.tests = self.preprocess(dtc)
        scores_ = []
        suite = TestSuite(dtc.tests)
        for t in suite:
            if 'RheobaseTest' in t.name: t.score_type = sciunit.scores.ZScore
            if 'RheobaseTestP' in t.name: t.score_type = sciunit.scores.ZScore
            if 'mean' not in t.observation.keys():
                t.observation['mean'] = t.observation['value']

            score_gene = t.judge(model)

            # All this nested uglyness can go away by making log_norm_score more robust.

            if not isinstance(type(score_gene),type(None)):
                if not isinstance(type(score_gene),sciunit.scores.InsufficientDataScore):
                    if not isinstance(type(score_gene.log_norm_score),type(None)):
                        try:

                            lns = np.abs(score_gene.log_norm_score)
                        except:
                            # works 1/2 time that log_norm_score does not work
                            # more informative than nominal bad score 100
                            lns = np.abs(score_gene.raw)
                    else:
                        # works 1/2 time that log_norm_score does not work
                        # more informative than nominal bad score 100

                        lns = np.abs(score_gene.raw)
                else:
                    pass

            scores_.append(lns)
        for i,s in enumerate(scores_):
            if s==np.inf:
                scores_[i] = 100 #np.abs(float(score_gene.raw))
        dtc.SA = ScoreArray(dtc.tests, scores_)

        if not hasattr(dtc,'gen'):
            dtc.gen = None
            dtc.gen = 1
        dtc.gen += 1
        assert dtc.SA is not None
        #dtc = self.get_agreement(dtc)

        return dtc


    #@timer
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
    #@timer
    def make_simulated_observations(self,original_test_dic,backend,random_param,dsolution=None):

        dtc = dsolution
        assert len(dtc.attrs)
        if self.protocol['elephant']:
            if 'protocol' in  original_test_dic.keys():
                original_test_dic.pop('protocol',None)
            if str('RheobaseTest') in original_test_dic.keys():
                dtc = get_rh(dtc,original_test_dic['RheobaseTest'])
                if type(dtc.rheobase) is type(dict()):
                    if dtc.rheobase['value'] is None:
                        return False, dtc
                elif type(dtc.rheobase) is type(float(0.0)):
                    pass

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
                    return (False,dtc)

            dtc.observation = simulated_observations

            try:
                dtc = self.pred_evaluation(dtc)


            except:
                return (False,dtc)
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
            dtc = dsolution

            if 'protocol' in  original_test_dic.keys():
                original_test_dic.pop('protocol',None)
            if str('RheobaseTest') in original_test_dic.keys():
                dtc = get_rh(dtc,original_test_dic['RheobaseTest'])
            if type(dtc.rheobase) is type(dict()):

                if dtc.rheobase['value'] is None:
                    return False, dtc
                elif type(dtc.rheobase) is type(float(0.0)):
                    pass

            xtests = list(copy.copy(original_test_dic).values())

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
            dtcpops = []
            for i in xargs:
                dtcpops.append(transform_delayed(i))
            dtcpop = dask.compute(*dtcpops)#[0]

            assert len(dtcpop) == len(pop)
            for dtc in dtcpop:
                dtc.backend = self.backend

            if self.hc is not None:
                for d in dtcpop:
                    d.attrs.update(self.hc)

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
            dtc = [ dask.compute(transform(xargs))[0] ]
            dtc.boundary_dict = None
            dtc.boundary_dict = self.boundary_dict
            return dtc

    #@timer
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
    def assert_against_zero_std(self,dtcpop,tests):

        for dtc in dtcpop:
            if not hasattr(dtc,'tests'):
                dtc.tests = self.tests
            if type(dtc.tests) is type(dict()):
                for t in dtc.tests.values():
                    assert 'std' in t.observation.keys()
                    #if float(t.observation['std']) == 0.0:
                        #t.observation['std'] = 0.1*copy.copy(t.observation['mean'])
                    #assert t.observation['std'] != 0.0

    #@timer
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

        if str('IZHI') in self.backend  or str('ADEXP') in self.backend:
            self.assert_against_zero_std(copy.copy(dtcpop),tests)

            dtcbag = [ delayed(dtc_to_rheo(d)) for d in dtcpop ]
            dtcpop = compute(*dtcbag)
            for p,d in zip(pop,dtcpop):
                p.rheobase = None
                p.rheobase = d.rheobase
            #(pop,dtcpop) = filtered(pop,dtcpop)
            dtcpop = list(map(self.format_test,dtcpop))
        else:
            print('multi-threading break?')
            dtcpop = [ dtc_to_rheo(d) for d in dtcpop ]
            #dtcpop = compute(*dtcbag)
            for p,d in zip(pop,dtcpop):
                p.rheobase = None
                p.rheobase = d.rheobase

            #(pop,dtcpop) = filtered(pop,dtcpop)
            dtcpop = list(map(self.format_test,dtcpop))

        for ind,d in zip(pop,dtcpop):
            if type(d.rheobase) is not type(None):
                if not hasattr(ind,'rheobase'):
                    ind.rheobase = None
                ind.rheobase = d.rheobase
                ind.dtc = d

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

    def seval(self,dtc):
        #dtc.get_agreement()
        dtc.self_evaluate()
        return dtc
    #@timer
    def parallel_route(self,pop,dtcpop,tests):
        td = self.td

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
            for dtc in dtcpop:
                if not hasattr(dtc,'tests'):
                    dtc.tests = copy.copy(self.tests)

                if isinstance(dtc.tests,type(dict())):
                    for t in dtc.tests.values():
                        assert 'std' in t.observation.keys()


            NPART = joblib.cpu_count()
            if self.PARALLEL_CONFIDENT:
                dtcbag = db.from_sequence(dtcpop,NPART)
                dtcpop = list(dtcbag.map(self.format_test).compute())
                dtcbag = db.from_sequence(dtcpop,NPART)
                dtcpop = list(dtcbag.map(self.seval).compute())
                #dtcpop = compute(*dtcbag)
                #with Parallel(n_jobs=joblib.cpu_count()) as parallel:
                #    dtcpop = Parallel(n_jobs=joblib.cpu_count())(delayed(self.seval)(d) for d in dtcpop[0:2])



            if self.MEMORY_FRIENDLY:# and self.backend is not str('IZHI'):
                passed = False
                lazy = []

                #for d in dtcpop:
                #   d = self.format_test_delayed(d)
                lazy = (dask.delayed(self.format_test)(d) for d in dtcpop)
                dtcpop = list(dask.compute(*lazy))

                #for d in dtcpop:
                #    d = dask.delayed(d.self_evaluate())
                #    lazy.append(d)
                #lazy = (dask.delayed(self.seval)(d) for d in dtcpop)

                #dtcpop = list(dask.compute(*lazy))

                #if self.backend is not "IZHI":
                #    dtcpop = Parallel(n_jobs=8)(delayed(self.seval)(d) for d in dtcpop)
                #else:
                lazy = (dask.delayed(self.seval)(d) for d in dtcpop)
                dtcpop = list(dask.compute(*lazy))

                #with Parallel(n_jobs=joblib.cpu_count) as parallel:
                #    dtcpop = Parallel(n_jobs=joblib.cpu_count)(delayed(self.elephant_evaluation)(d) for d in dtcpop[0:2])
                # smaller_pop = [d for d in dtcpop for _,fitness in d.SA.values if fitness != 100 ]
                # pop, dtcpop = self.make_up_lost(pop,smaller_pop,self.td)
                for d in dtcpop:
                    assert hasattr(d, 'tests')
                    assert d.SA is not None
                #for d in dtcpop:
                #    d.tests = copy.copy(self.tests)

            if not self.PARALLEL_CONFIDENT and not self.MEMORY_FRIENDLY:
                for d in dtcpop:
                    assert d.rheobase is not None
                dtcpop = list(map(self.format_test,dtcpop))
                dtcpop = list(map(self.elephant_evaluation,dtcpop))
                for d in dtcpop:
                    assert hasattr(d, 'tests')

        return pop, dtcpop


    def test_runner(self,pop,td,tests):
        if self.protocol['elephant']:
            if type(tests) is type(dict()):
                for t in tests.values():
                    assert 'std' in t.observation.keys()

            elif type(tests) is type(list()):
                for t in tests:
                    assert 'std' in t.observation.keys()
            if self.MU is None:
                self.MU = len(pop)
                #print(self.MU)
            pop_, dtcpop = self.obtain_rheobase(pop, tests)
            pop, dtcpop = self.make_up_lost(copy.copy(pop_), dtcpop, td)

            if hasattr(self,'exhaustive'):
                # there are many models, which have no actual rheobase current injection value.
                # filter, filters out such models,
                # gew genes, add genes to make up for missing values.
                # delta is the number of genes to replace.
                pop,dtcpop = self.parallel_route(pop, dtcpop, tests)#, clustered=False)
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
        self.assert_against_zero_std(dtcpop,tests)
        pop,dtcpop = self.parallel_route(pop, dtcpop, tests)#, clustered=False)
        both = [(ind,dtc) for ind,dtc in zip(pop,dtcpop) if dtc.SA is not None]
        for ind,dtc in both:
            ind.dtc = None
            ind.dtc = dtc
            if dtc.SA is not None:
                ind = both[0][0]
                dtc = both[0][1]

            if not hasattr(ind,'fitness'):
                ind.fitness = pop_[0].fitness
                for i,v in enumerate(list(ind.fitness.values)):
                    ind.fitness.values[i] = list(ind.dtc.SA.values())[i]

        pop = [ ind for ind,dtc in zip(pop,dtcpop) if dtc.SA is not None ]
        dtcpop = [ dtc for ind,dtc in zip(pop,dtcpop) if dtc.SA is not None ]
        return pop,dtcpop

    #@timer
    def make_up_lost(self,pop,dtcpop,td):
        '''
        make new genes, actively replace genes for unstable solutions.
        Alternative: let gene pool shrink momentarily, risks in breading.
        '''

        spare = copy.copy(dtcpop)
        before = len(pop)
        (pop,dtcpop) = filtered(pop,dtcpop)
        after = len(pop)
        delta = before-after
        if not delta:


            return pop, dtcpop
        if delta:
            '''
            if delta<len(pop):
                pop.extend(pop[0:delta])
                dtcpop.extend(dtcpop[0:delta])

                return pop, dtcpop
            elif len(pop)>1:
                for i in range(0,delta):
                    pop.extend(pop[0:len(pop)])
                    dtcpop.extend(dtcpop[0:len(pop)])
                return pop, dtcpop


            elif len(pop)==1:
                for i in range(0,delta):
                    pop.append(pop[0])
                    dtcpop.append(dtcpop[0])
                print(len(pop))
                return pop, dtcpop



            else:
            '''
            cnt = 0
            while delta:
                pop_,dtcpop_ = self.boot_new_genes(delta,spare)
                for dtc,ind in zip(pop_,dtcpop_):
                    ind.from_imputation = None
                    ind.from_imputation = True
                    ind.valid = False
                    dtc.from_imputation = True

                    if len(spare)>1:
                        sp = spare[0]
                    else:
                        sp = spare
                    dtc.tests = copy.copy(sp.tests)


                    for i,t in enumerate(dtc.tests[0:7]):
                        k = t.name
                        try:
                            assert dtc.tests[i].observation['mean'] == self.tests[k].observation['mean']
                        except:
                            print(dtc.tests[i].observation['mean'])
                    ind.dtc = dtc

                pop_ = [ p for p in pop_ if len(p)>1 ]

                pop.extend(pop_)
                dtcpop.extend(dtcpop_)
                (pop,dtcpop) = filtered(pop,dtcpop)
                for i,p in enumerate(pop):
                    if not hasattr(p,'fitness'):
                        p.fitness = fitness_attr

                after = len(pop)
                delta = self.MU-after
                if delta:
                    continue
                else:

                    break
            return pop, dtcpop
    #@timer
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
            fitnesses = [ list(dtc.SA.values()) for dtc in dtcpop ]
            return fitnesses, pop, dtcpop


        return pop
    #@timer
    def boot_new_genes(self,number_genes,dtcpop):
        '''
        Boot strap new genes to make up for completely called onesself.
        '''
        from neuronunit.optimization.optimizations import SciUnitoptimization

        random.seed(datetime.now())
        DO = SciUnitoptimization(MU = number_genes,
                                 tests = self.tests, boundary_dict =self.boundary_dict,
                                 backend = self.backend,protocol = self.protocol)#,, boundary_dict = ss, elite_size = 2, hc=hc)

        #DO.setnparams(nparams = len(dtcpop[0].attrs), boundary_dict = self.boundary_dict)
        if self.td is None:
            self.td = list(OrderedDict(self.boundary_dict).keys())
        DO.setnparams(nparams = len(self.td), boundary_dict = self.boundary_dict)
        DO.setup_deap()
        if 1==number_genes:
            # you asked for one gene but I will give you 5 assuming at least
            # one in 5 has a stable rheobase.
            # extras disgarded below.
            # price of genes very cheap.
            pop = DO.set_pop(boot_new_random=5)
        if 1<number_genes and number_genes<5:
            pop = DO.set_pop(boot_new_random=5)

        else:
            pop = copy.copy(DO.set_pop(boot_new_random=number_genes))
        dtcpop_ = self.update_dtc_pop(pop)
        dtcbag = [ delayed(dtc_to_rheo(d)) for d in dtcpop_ ]
        dtcpop = compute(*dtcbag)
        for i,ind in enumerate(pop):
            #dtcpop_[i].self_evaluate()
            pop[i].rheobase = dtcpop_[i].rheobase
            pop[i].dtc = dtcpop_[i]
        pop = pop[0:number_genes]
        dtcpop_ = dtcpop_[0:number_genes]
        del DO
        #fitnesses = list(map(self.evaluate,dtcpop_))
        #for p,f in zip(fitnesses,pop):
        #    p.fitness = f

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
