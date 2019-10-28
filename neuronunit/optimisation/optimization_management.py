# Its noAAAt that this file is responsible for doing plotting,
# but it calls many modules that are, such that it needs to pre-empt

# setting of an appropriate backend.
# optional imports
import warnings
import matplotlib
try:
    matplotlib.use('agg')
except:
    warnings.warn('X11 plotting backend not available, consider installing')
#_arraytools
SILENT = True
RATIO_SCORE = False
if SILENT:
    warnings.filterwarnings("ignore")

# optional imports

CONFIDENT = True
#    Goal is based on this. Don't optimize to a singular point, optimize onto a cluster.
#    Golowasch, J., Goldman, M., Abbott, L.F, and Marder, E. (2002)
#    Failure of averaging in the construction
#    of conductance-based neuron models. J. Neurophysiol., 87: 11291131.
#from neuronunit.tests.elephant_tests import ETest

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


from neuronunit.capabilities.spike_functions import get_spike_waveforms

import logging
logger = logging.getLogger('brian2')
#logger.debug('test')
import copy
import math
import quantities as pq
import numpy

#from pyneuroml import pynml

from neuronunit.optimisation.data_transport_container import DataTC
#from neuronunit.models.interfaces import glif

# Import get_neab has to happen exactly here. It has to be called only on
#from neuronunit import tests
#from neuronunit.models.reduced import ReducedModel
from neuronunit.optimisation.model_parameters import path_params
from neuronunit.optimisation import model_parameters as modelp
from itertools import repeat
from neuronunit.tests.base import AMPL, DELAY, DURATION
from neuronunit.models import ReducedModel
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from collections.abc import Iterable
DEBUG = False
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
import os
import neuronunit
anchor = neuronunit.__file__
anchor = os.path.dirname(anchor)

mypath = os.path.join(os.sep,anchor,'tests/russell_tests.p')
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


# Helper tests are dummy instances of NU tests.
# They are used by other methods analogous to a base class,
# these are base instances that become more derived
# contexts, that modify copies of the helper class in place.
rts,complete_map = pickle.load(open(mypath,'rb'))
df = pd.DataFrame(rts)
for key,v in rts.items():
    helper_tests = [value for value in v.values() ]
    break

class TSD(dict):
    def __init__(self,tests=[],use_rheobase_score=False):
       super(TSD,self).__init__()
       self.update(tests)
       self.use_rheobase_score=use_rheobase_score


    def optimize(self,param_edges,backend=None,protocol={'allen': False, 'elephant': True},MU=5,NGEN=5):
        from neuronunit.optimisation.optimisations import run_ga
        ga_out,DO = run_ga(param_edges, NGEN, self, free_params=param_edges.keys(), \
                           backend=backend, MU = 8,  protocol=protocol)
        # dtc_pop = self.update_dtc_pop(ga_out['pf'], DO.OM.td)
        if not hasattr(ga_out['pf'][0],'dtc') and 'dtc_pop' not in ga_out.keys():
            _,dtc_pop = DO.OM.test_runner(ga_out['pf'],DO.OM.td,DO.OM.tests)
            ga_out['dtc_pop1'] = dtc_pop

        return ga_out, DO


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

'''depriciated

def _and_dont_plot(dtc):
    # For debugging backends during development.
    model = mint_generic_model(dtc.backend)

    try:
        rheobase = dtc.rheobase['value']
    except:
        rheobase = dtc.rheobase

    uc = { 'injected_square_current': {'amplitude':rheobase,'duration':DURATION,'delay':DELAY }}

    #dtc.run_number += 1
    model.set_attrs(**dtc.attrs)
    model.inject_square_current(uc['injected_square_current'])
    return model, model.get_membrane_potential().times,model.get_membrane_potential(),uc
'''


#
@cython.boundscheck(False)
@cython.wraparound(False)
def inject_and_plot(dtc,second_pop=None,third_pop=None,figname='problem',snippets=False,experimental_cell_type="neo_cortical"):
    sns.set_style("darkgrid")

    if not isinstance(dtc, Iterable):
        model = mint_generic_model(dtc.backend)
        if hasattr(dtc,'rheobase'):
            try:
                rheobase = dtc.rheobase['value']
            except:
                rheobase = dtc.rheobase
                uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
        if hasattr(dtc,'ampl'):
            uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

        model.set_attrs(**dtc.attrs)
        model.inject_square_current(uc)
        if str(dtc.backend) is str('ADEXP'):
            model.finalize()
        else:
            pass
        #vm = model.get_membrane_potential().magnitude
        sns.set_style("darkgrid")
        plt.plot(model.get_membrane_potential().times,model.get_membrane_potential().magnitude)#,label='ground truth')
        plot_backend = mpl.get_backend()
        if plot_backend == str('agg'):
            plt.savefig(figname+str('debug.png'))
        else:
            plt.show()

    else:

        if type(second_pop) is not type(None):
            dtcpop = copy.copy(dtc)
            dtc = None

            fig = plt.figure(figsize=(11,11),dpi=100)
            ax = fig.add_subplot(111)


            for dtc in dtcpop:
                model = mint_generic_model(dtc.backend)
                if hasattr(dtc,'rheobase'):
                    try:
                        rheobase = dtc.rheobase['value']
                    except:
                        rheobase = dtc.rheobase
                        uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                if hasattr(dtc,'ampl'):
                    uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}
                #dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                if rheobase is None:
                    break
                model.inject_square_current(uc)
                #if model.get_spike_count()>1:
                #    break
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    pass
                vm = model.get_membrane_potential()#.magnitude
                if str("RAW") in dtc.backend:
                    label=str('Izhikevich Model')

                if str("ADEXP") in dtc.backend:
                    label=str('Adaptive Exponential Model')
                if str("GLIF") in dtc.backend:
                    label=str('Generalized Leaky Integrate and Fire')

                sns.set_style("darkgrid")
                if snippets:
                    snippets_ = get_spike_waveforms(vm)
                    plt.plot(snippets_.times,snippets_,color='red',label=str('model type: ')+label)#,label='ground truth')
                else:
                    plt.plot(vm.times,vm,color='red',label=str('model type: ')+label)#,label='ground truth')
                ax.legend()
                sns.set_style("darkgrid")

                plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')

            for dtc in second_pop:
                model = mint_generic_model(dtc.backend)
                if hasattr(dtc,'rheobase'):
                    try:
                        rheobase = dtc.rheobase['value']
                    except:
                        rheobase = dtc.rheobase
                        uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                if hasattr(dtc,'ampl'):
                    uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

                #uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                #dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                #if model.get_spike_count()>1:
                #    break
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    pass
                vm = model.get_membrane_potential()#.magnitude
                if str("RAW") in dtc.backend:
                    label=str('Izhikevich Model')

                if str("ADEXP") in dtc.backend:
                    label=str('Adaptive Exponential Model')

                if str("GLIF") in dtc.backend:
                    label=str('Generalized Leaky Integrate and Fire')
                #label = label+str(latency)

                sns.set_style("darkgrid")
                if snippets:
                    snippets_ = get_spike_waveforms(vm)
                    plt.plot(snippets_.times,snippets_,color='blue',label=str('model type: ')+label)#,label='ground truth')
                else:
                    plt.plot(vm.times,vm,color='blue',label=str('model type: ')+label)#,label='ground truth')
                #plt.plot(model.get_membrane_potential().times,vm,color='blue',label=label)#,label='ground truth')
                #ax.legend(['A simple line'])
                ax.legend()
                sns.set_style("darkgrid")

                plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')

            if third_pop is None:
                pass
            else:

                for dtc in third_pop:
                    model = mint_generic_model(dtc.backend)
                    if hasattr(dtc,'rheobase'):
                        try:
                            rheobase = dtc.rheobase['value']
                        except:
                            rheobase = dtc.rheobase
                            uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                    if hasattr(dtc,'ampl'):
                        uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

                    model.set_attrs(**dtc.attrs)
                    model.inject_square_current(uc)
                    #if model.get_spike_count()>1:
                    #    break
                    if str(dtc.backend) in str('ADEXP'):
                        model.finalize()
                    else:
                        pass
                    vm = model.get_membrane_potential()#.magnitude
                    sns.set_style("darkgrid")

                    if str("RAW") in dtc.backend:
                        label=str('Izhikevich Model')

                    if str("ADEXP") in dtc.backend:
                        label=str('Adaptive Exponential Model')
                    if str("GLIF") in dtc.backend:
                        label=str('Generalized Leaky Integrate and Fire')
                    #label = label+str(latency)
                    if snippets:

                        snippets_ = get_spike_waveforms(vm)
                        plt.plot(snippets_.times,snippets_,color='green',label=str('model type: ')+label)#,label='ground truth')
                        #ax.legend(['A simple line'])
                        ax.legend()
                    else:
                        plt.plot(vm.times,vm,color='green',label=str('model type: ')+label)#,label='ground truth')

                plot_backend = mpl.get_backend()
                sns.set_style("darkgrid")

                plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')

                if str('agg') in plot_backend:

                    plt.savefig(figname+str('all_traces.png'))
                else:
                    plt.show()
        else:
            dtcpop = copy.copy(dtc)
            dtc = None
            for dtc in dtcpop[0:2]:
                model = mint_generic_model(dtc.backend)
                if hasattr(dtc,'rheobase'):
                    try:
                        rheobase = dtc.rheobase['value']
                    except:
                        rheobase = dtc.rheobase
                        uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                if hasattr(dtc,'ampl'):
                    uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

                #dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                vm = model.get_membrane_potential().magnitude
                sns.set_style("darkgrid")

                plt.plot(model.get_membrane_potential().times,vm)#,label='ground truth')
            plot_backend = mpl.get_backend()
            sns.set_style("darkgrid")

            plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude (mV)')

            if plot_backend == str('agg'):
                plt.savefig(figname+str('all_traces.png'))
            else:
                plt.show()
    return None, None

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
        #import pdb
        #pdb.set_trace()
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
    if 'RAW' in str(backend):
        random_param.pop('Iext',None)
        rp = {}
        chosen_keys =[str('a'),str('b'),str('c'),str('C'),str('d')]
        for key in chosen_keys:
            rp[key] = random_param[key]
    if str('ADEXP') in str(backend):
        random_param.pop('Iext',None)
        rp = random_param
        chosen_keys = rp.keys()

    if 'GLIF' in str(backend):
        random_param['init_AScurrents'] = [0.0,0.0]
        random_param['asc_tau_array'] = [0.3333333333333333,0.01]
        rp = random_param
        chosen_keys = rp.keys()

    dsolution = DataTC()
    dsolution.attrs = rp
    #import pdb
    #pdb.set_trace()
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

def get_centres(use_test,backend,explore_param):
    '''
    Do optimization, but then get cluster centres.
    '''
    MU = 7
    NGEN = 7
    test_opt = {}
    for index,test in enumerate(use_test):
        ga_out, DO = run_ga(explore_param,NGEN,test,free_params=free_params, NSGA = True, MU = MU,backed=backend, selection=str('selNSGA2'))
        td = DO.td # td, transfer dictionary, a consistent, stable OrderedDict of parameter values.
        test_opt[test.name] = ga_out
    with open('qct.p','wb') as f:
        pickle.dump(test_opt,f)

    all_val = {}
    for key,value in test_opt.items():
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
    LEMS_MODEL_PATH = path_params['model_path']
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = str(backend))
    return model


def save_models_for_justas(dtc):
    with open(str(dtc.attrs)+'.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)

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
    #obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(**dtc.attrs)
    if test.passive:
        test.setup_protocol(model)
        try:
            pred = test.extract_features(model,test.get_result(model))
        except:
            pred = None
    else:
        pred = test.generate_prediction(model)
    return pred

'''
#def t2m(bridge_judge):
def _pseudo_decor(bridge_judge):
    (test, dtc) = bridge_judge.test_and_dtc
    obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(**dtc.attrs)
    return model
'''
#from functools import partial
#t2m = partial(_pseudo_decor, argument=arg)

#@_pseudo_decor
def bridge_judge(test_and_dtc):
    (test, dtc) = test_and_dtc
    obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(**dtc.attrs)


    if test.passive:
        test.setup_protocol(model)
        pred = test.extract_features(model,test.get_result(model))
        # pred = tests[0].extract_features(dtc.dtc_to_model(),tests[0].get_result(dtc.dtc_to_model()))

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
    #if 'mean' in pred.keys():
    #    dtc.predictions[test.name]['value'] = pred['val'']

    #if 'value' in dtc.observations.keys():
        #    dtc.observations.pop('value',None)

    if type(pred) is not type(None):
        #try:
        score = test.compute_score(test.observation,pred)
        #except:
        #    score = None
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

            #import pdb; pdb.set_trace()
            score = test.compute_score(test.observation,width_obs)
            return score, dtc

        elif str('InjectedCurrentAPAmplitudeTest') in test.name:
            #import pdb; pdb.set_trace()

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
    #if type(rtest) is not None:
    #    return rtest
    if 'RAW' in backend_ or 'HH' in backend_:

        rtest = RheobaseTest(observation=place_holder,
                             name='RheobaseTest')
    elif 'ADEXP' in backend_ or 'GLIF' in backend_:#Backend::
        rtest = RheobaseTestP(observation=place_holder,
                                name='RheobaseTest')
    else:
        rtest = RheobaseTest(observation=place_holder,
                         name='RheobaseTest')
    dtc.rheobase = None
    model = mint_generic_model(backend_)
    model.set_attrs(**dtc.attrs)
    rtest.params['injected_square_current'] = {}
    rtest.params['injected_square_current']['delay'] = DELAY
    rtest.params['injected_square_current']['duration'] = DURATION
    dtc.rheobase = rtest.generate_prediction(model)
    if dtc.rheobase is None:
        dtc.rheobase = - 1.0
    return dtc

'''
def dtc_to_rheo_serial(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    dtc.evaluate = {}
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    rtest = [ t for t in dtc.tests if str('RheobaseTest') == t.name ]
    if len(rtest):
        rtest = rtest[0]
        dtc.rheobase = rtest.generate_prediction(model)
        if type(dtc.rheobase) is not type(None):# and dtc.rheobase !=-1.0:
            dtc.rheobase = dtc.rheobase['value']
            obs = rtest.observation
            score = rtest.compute_score(obs,dtc.rheobase)
            dtc.evaluate[rtest.name] = 1.0 - score.norm_score
            rtest.params['injected_square_current']['amplitude'] = dtc.rheobase
        else:
            dtc.rheobase = None
            dtc.evaluate[rtest.name] = 1.0

    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
    return dtc
'''

def substitute_parallel_for_serial(rtest):
    rtest = RheobaseTestP(rtest.observation)
    return rtest

def get_rtest(dtc):
    place_holder = {'n': 86, 'mean': 10 * pq.pA, 'std': 10 * pq.pA, 'value': 10 * pq.pA}

    if not hasattr(dtc,'tests'):#, type(None)):
        if 'RAW' in dtc.backend or 'HH' in dtc.backend:# or 'GLIF' in dtc.backend:#Backend:
            rtest = RheobaseTest(observation=place_holder,
                                    name='RheobaseTest')
        else:
            rtest = RheobaseTestP(observation=place_holder,
                                    name='RheobaseTest')
    else:
        if 'RAW' in dtc.backend or 'HH' in dtc.backend:# or 'GLIF' in dtc.backend:
            if not isinstance(dtc.tests, Iterable):
                rtest = dtc.tests
            else:
                #try:
                rtest = [ t for t in dtc.tests if hasattr(t,'name') ]
                if len(rtest):
                    rtest = [ t for t in rtest if str('RheobaseTest') == t.name ]
                    if len(rtest):
                        rtest = rtest[0]
                    else:
                        rtest = RheobaseTest(observation=place_holder,
                                                name='RheobaseTest')
                    #rtest = substitute_parallel_for_serial(rtest[0])
                else:
                    rtest = RheobaseTest(observation=place_holder,
                                            name='RheobaseTest')

        else:
            if not isinstance(dtc.tests, Iterable):
                rtest = dtc.tests
            else:
                rtest = [ t for t in dtc.tests if hasattr(t,'name') ]
                if len(rtest):
                    rtest = [ t for t in rtest if str('RheobaseTestP') == t.name ]
                    if len(rtest):
                        rtest = substitute_parallel_for_serial(rtest[0])
                    else:
                        rtest = RheobaseTestP(observation=place_holder,
                                                name='RheobaseTest')
                else:
                    rtest = RheobaseTestP(observation=place_holder,
                                            name='RheobaseTest')

    return rtest

#from collections import OrderedDict
def to_map(params,bend):
    dtc = DataTC()
    dtc.attrs = params
    dtc.backend = bend
    dtc = dtc_to_rheo(dtc)
    return dtc

def dtc_to_model(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    if not hasattr(dtc,'scores'):
        dtc.scores = None
    if type(dtc.scores) is type(None):
        dtc.scores = {}
    model = mint_generic_model(dtc.backend)
    model.attrs = dtc.attrs
    return model

def dtc_to_rheo(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    if not hasattr(dtc,'scores'):
        dtc.scores = None
    if type(dtc.scores) is type(None):
        dtc.scores = {}
    model = mint_generic_model(dtc.backend)
    model.set_attrs(**dtc.attrs)
    if hasattr(dtc,'tests'):
        if type(dtc.tests) is type({}) and str('RheobaseTest') in dtc.tests.keys():
            rtest = dtc.tests['RheobaseTest']
        else:
            rtest = get_rtest(dtc)
    else:
        rtest = get_rtest(dtc)

    if rtest is not None:
        if isinstance(rtest,Iterable):
            rtest = rtest[0]
        dtc.rheobase = rtest.generate_prediction(model)
        if dtc.rheobase is not None:
            if type(dtc.rheobase['value']) is not type(None):
                if not hasattr(dtc,'prediction'):
                    dtc.prediction = {}
                dtc.prediction[str(rtest.name)] = dtc.rheobase
                dtc.rheobase = dtc.rheobase['value']

                obs = rtest.observation
                rtest.prediction = None
                rtest.prediction = dtc.rheobase
                #import pdb
                #pdb.set_trace()
                score = rtest.compute_score(obs,dtc.rheobase)
                if type(score.norm_score) is not type(None):
                    dtc.scores[rtest.name] = 1.0 - float(score.norm_score)
                else:
                    dtc.scores[rtest.name] = 1.0
            else:
                dtc.rheobase = None
                dtc.scores[rtest.name] = 1.0
        else:
            dtc.rheobase = None
            dtc.scores[rtest.name] = 1.0

    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
    return dtc


def score_proc(dtc,t,score):
    dtc.score[str(t)] = {}
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

def switch_logic(xtests):
    # move this logic into sciunit tests
    '''
    Hopefuly depreciated by future NU debugging.
    '''
    if str("TSD") in str(type(xtests)):
        xtests = list(xtests.values())
    if not isinstance(xtests,Iterable):
        pass
    else:
        for t in xtests:
            try:
                t.passive = None
                t.active = None
            except:
                import pdb
                pdb.set_trace()
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
'''depricated
def allen_scores(dtc):
    ephys_dict = dtc.ephys.as_dict()
    if not 'spikes' in ephys_dict.keys():
        dtc.scores = None
        return dtc
    dtc.scores = None
    dtc.scores = {}
    for k,v in compare.items():
        dtc.scores[k] = 1.0
    helper = helper_tests[0]

    for k,observation in compare.items():
        if str(k) not in str('spikes'):
            #compute interspike firing frame_dynamics

            obs = {}
            if type(observation) is not type({'1':0}):
                obs['mean'] = observation
            else:
                obs = observation
            prediction = {}
            if k in ephys_dict.keys():
                assert prediction['mean'] == ephys_dict[k]['mean']
                #helper.name = str(k)
                #obs['std']= 1.0
                #prediction['std']=1.0
                prediction = dtc.preds[k]
                try:
                    score = VmTest.compute_score(helper,obs,prediction)
                except:

                    score = None
                dtc.tests[k] = VmTest(obs)#.compute_score(helper,obs,prediction)
                if score is not None and score.norm_score is not None:
                    dtc.scores[k] = 1.0-score.norm_score
                else:
                    dtc.scores[k] = 1.0
        if str(k) in str('spikes'):
            #compute perspike waveform features on just the first spike
            try:
                first_spike = observation[0]
            except:
                pass
            for key,spike_obs in first_spike.items():

                #if not str('direct') in key and not str('adp_i') in key and not str('peak_i') in key and not str('fast_trough_i') and not str('fast_trough_i') and not str('trough_i'):
                try:
                    obs = {'mean': spike_obs}
                    prediction = {'mean': ephys_dict['spikes'][0][key]}
                    helper.name = str(key)
                    obs['std']=1.0
                    prediction['std']=1.0
                    dtc.preds[key] = prediction
                    #score = VmTest.compute_score(helper,obs,prediction)

                    try:
                        score = VmTest.compute_score(helper,obs,prediction)
                    except:
                        score = None

                    dtc.tests[key] = VmTest(obs)
                    if not score is None and not score.norm_score is None:
                        dtc.scores[key] = 1.0-score.norm_score
                except:
                    dtc.scores[key] = 1.0
    dtc.ephys = None
    dtc.ephys = ephys
    dtc.spike_number = len(ephys_dict['spikes'])
    spike_cnt_pred = model.get_spike_count()
    try:
        delta = spike_cnt_pred-target
        dtc.scores['spk_count'] = np.abs(delta)#sigmoid()#spike_cnt_pred-target #

    except:
        pass
    try:
        delta = np.abs(float(dtc.ampl)-dtc.pre_obs['current_test'])
        dtc.scores['current_test'] = delta #sigmoid(np.abs(delta))

    except:
        pass

    return dtc
'''
def get_dtc_pop(result_ADEXP,filtered_tests,model_parameters,backend = 'ADEXP'):
    from neuronunit.optimisation.optimisations import SciUnitOptimisation
    from neuronunit.optimisation.optimization_management import get_dtc_pop
    import copy

    random.seed(64)
    boundary_dict=model_parameters.MODEL_PARAMS[backend]
    tests=filtered_tests['Hippocampus CA1 basket cell']
    try:
        DO = SciUnitOptimisation(offspring_size = 1,
            error_criterion = tests, boundary_dict = boundary_dict,
                                     backend = backend, selection = str('selNSGA'))#,simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)
    except:
        DO = SciUnitOptimisation(offspring_size = 1,
            error_criterion = tests, boundary_dict = boundary_dict,
                                     backend = backend, selection = str('selNSGA'))#,simulated_obs=dtc.preds)#,, boundary_dict = ss, elite_size = 2, hc=hc)

    DO.setnparams(nparams = len(result_ADEXP['ADEXP']['olf']), boundary_dict = boundary_dict)
    DO.setup_deap()

    dtcdic = {}
    for k,v in result_ADEXP[backend].items():

        dtcpop = []
        for i in v:
            dtcpop.append(transform((i,DO.td,backend)))
            dtcpop[-1].backend = backend
            dtcpop[-1] = dtc_to_rheo(dtcpop[-1])
        dtcdic[k] = copy.copy(dtcpop)
    return dtcdic, DO


def just_allen_predictions(dtc):
    current = {'injected_square_current':
                {'amplitude':dtc.ampl, 'delay':DELAY, 'duration':DURATION}}
    comp = False
    if hasattr(dtc,'pre_obs'):

        if type(dtc.pre_obs) is not type(None):
            compare = dtc.pre_obs
            comp = True
            if 'spk_count' not in compare.keys():
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
        spks = ft.detect_putative_spikes(np.array(v),np.array(t))
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
        prediction = {}
        dtc.preds= {}
        obs= {}
        for k in ephys_dict.keys():
            if 'spikes' not in k:
                prediction['mean'] = ephys_dict[k]
                prediction['std'] = 10.0
                dtc.preds[k] = prediction

            else:
                for other in ephys_dict['spikes'][0].keys():
                    temp = ephys_dict['spikes'][0][other]
                    prediction['mean'] = temp
                    prediction['std'] = 10.0
                    dtc.preds[other] = prediction
                dtc.spike_cnt = len(ephys_dict['spikes'])
                dtc.preds['spikes'] = dtc.spike_cnt

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
    helper = helper_tests[0]
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
                obs['std'] = obs['mean']/15.0
                dtc.preds[k] = prediction
                prediction['std'] = prediction['mean']/15.0

                try:
                    score = VmTest.compute_score(helper,obs,prediction)
                except:
                    score = None
                dtc.tests[k] = VmTest(obs)#.compute_score(helper,obs,prediction)
                if score is not None and score.norm_score is not None:
                    dtc.scores[k] = 1.0-float(score.norm_score)
                else:
                    dtc.scores[k] = 1.0

        '''
        compute perspike waveform features on just the first spike
        '''
        first_spike = ephys_dict['spikes'][0]
        first_spike.pop('direct',None)
        for key,spike_obs in first_spike.items():

            #if not str('direct') in key and not str('adp_i') in key and not str('peak_i') in key and not str('fast_trough_i') and not str('fast_trough_i') and not str('trough_i'):
            #try:
            obs = {'mean': compare[key]['mean']}

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
            if not score is None and not score.norm_score is None:
                dtc.scores[key] = 1.0-score.norm_score
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
depricated
def allen_features_block2(dtc):
    current = {'injected_square_current':
                {'amplitude':dtc.ampl, 'delay':DELAY, 'duration':DURATION}}
    compare = dtc.pre_obs


    model = dtc.dtc_to_model()
    model.set_attrs(dtc.attrs)
    assert model is not None
    #model.set_attrs(**dtc.attrs)
    vm30 = model.inject_square_current(current['injected_square_current'])
    vm30 = model.get_membrane_potential()

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
        spks = ft.detect_putative_spikes(np.array(v),np.array(t))
        ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
        ephys.process_spikes()

    except:
        #rectify unfilterable high sample frequencies by downsampling them
        #downsample too densely sampled signals.
        #Making them amenable to Allen analysis

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

    if not 'spikes' in ephys_dict.keys():
        dtc.scores = None
        dtc.preds = {}
        return dtc
    dtc.scores = None
    dtc.scores = {}
    for k,v in compare.items():
        dtc.scores[k] = 1.0
    helper = helper_tests[0]

    dtc.preds = {}
    dtc.tests = {}
    for k,observation in compare.items():
        if str(k) not in str('spikes'):
            #compute interspike firing frame_dynamics

            obs = {}
            if type(observation) is not type(dict()):
                obs['mean'] = observation
            else:
                obs = observation
            prediction = {}
            if k in ephys_dict.keys():
                prediction['mean'] = ephys_dict[k]
                helper.name = str(k)
                obs['std'] = 10.0
                prediction['std'] = 10.0
                dtc.preds[k] = prediction
                try:
                    score = VmTest.compute_score(helper,obs,prediction)
                except:
                    #print(helper,obs,prediction)
                    score = None
                dtc.tests[k] = VmTest(obs)#.compute_score(helper,obs,prediction)
                if score is not None and score.norm_score is not None:
                    dtc.scores[k] = 1.0-float(score.norm_score)
                else:
                    dtc.scores[k] = 1.0
        if str(k) in str('spikes'):
            #compute perspike waveform features on just the first spike
            first_spike = ephys_dict[k][0]

            for key,spike_obs in first_spike.items():

                #if not str('direct') in key and not str('adp_i') in key and not str('peak_i') in key and not str('fast_trough_i') and not str('fast_trough_i') and not str('trough_i'):
                try:
                    obs = {'mean': spike_obs}
                    prediction = {'mean': ephys_dict['spikes'][0][key]}
                    helper.name = str(key)
                    obs['std']=10.0
                    prediction['std']=10.0
                    dtc.preds[key] = prediction
                    #score = VmTest.compute_score(helper,obs,prediction)

                    try:
                        score = VmTest.compute_score(helper,obs,prediction)
                    except:
                        score = None
                    dtc.tests[key] = VmTest(obs)
                    if not score is None and not score.norm_score is None:
                        dtc.scores[key] = 1.0-score.norm_score
                    else:
                        dtc.scores[key] = 1.0
                except:
                    dtc.scores[key] = 1.0
    dtc.ephys = None
    dtc.ephys = ephys
    dtc.spike_number = len(ephys_dict['spikes'])
    spike_cnt_pred = model.get_spike_count()
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
    model = mint_generic_model(dtc.backend)
    model.set_attrs(**dtc.attrs)
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
            #print(target_current)
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
    model = mint_generic_model(dtc.backend)
    model.set_attrs(**dtc.attrs)
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
    dtc.dm_test_features = None
    dtc.dm_test_features = dm_test_features
    return dtc

def nuunit_dm_rheo_evaluation(dtc):
    model = mint_generic_model(dtc.backend)
    model.set_attrs(**dtc.attrs)
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
    for int_,t in enumerate(dtc.ascores.keys()):
       if regularization:
          if dtc.ascores[str(t)] is None:
              fitness[int_] = 1.0
          else:
              fitness[int_] = dtc.ascores[str(t)]**(1.0/2.0)
       else:
          if dtc.ascores[str(t)] is None:
              fitness[int_] = 1.0
          else:
              fitness[int_] = dtc.ascores[str(t)]
    return tuple(fitness,)

def evaluate(dtc,regularization=False):
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    greatest = len(dtc.tests)
    fitness = []# 1.0 for i in range(0,greatest) ]

    if not hasattr(dtc,str('scores')):
        return fitness

    #fitness = [ 1.0 for i in range(0,greatest) ]
    for int_,t in enumerate(dtc.scores.keys()):
       if regularization:
          fitness.append(float(dtc.scores[str(t)]**(1.0/2.0)))
       else:
          fitness.append(float(dtc.scores[str(t)]))
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
''' Depricated
def update_dtc_pop(self,pop, td):
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    if self.backend is not None:
        _backend = pop[0].backend

    if isinstance(pop, Iterable):# and type(pop[0]) is not type(str('')):
        xargs = zip(pop,repeat(td),repeat(_backend))
        npart = np.min([multiprocessing.cpu_count(),len(pop)])
        bag = db.from_sequence(xargs, npartitions = npart)
        dtcpop = list(bag.map(transform).compute())

        assert len(dtcpop) == len(pop)
        for dtc in dtcpop:
            dtc.boundary_dict = None
            dtc.boundary_dict = pop[0].boundary_dict
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
        dtc.boundary_dict = None
        dtc.boundary_dict = pop[0].boundary_dict
        return dtc

'''

def scale(X):
    before = copy.copy(X)
    for i in range(0,np.shape(X)[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
    return X, before

def data_versus_optimal1(dtc_pop):
    rts,complete_map = pickle.load(open('../tests/russell_tests.p','rb'))

    #dtcpop = [ p.dtc for p in ga_out['pf'] ]
    #pop = [ p for p in ga_out['pf'] ]
    # first a nice violin plot of the test data.
    to_norm = np.matrix([list(t.data) for t in rts ])

    X,before = scale(to_norm)

    ax = sns.violinplot(x="test type", y="physical unit", hue="smoker",
                 data=X, palette="muted")

    for t in tests:
        plt.clf()
        fig, ax = plt.subplots()
        if t.name not in ga_out['pf'][0].dtc.prediction.keys():
            try:
                pred = ga_out['pf'][0].dtc.prediction['RheobaseTestP']

            except:
                pred = ga_out['pf'][0].dtc.rheobase
            if not isinstance(pred, dict):
                pred = {'value':pred}
        else:
            pred = ga_out['pf'][0].dtc.prediction[t.name]
        try:
            opt_value = pred['value']
        except:
            opt_value = pred['mean']
        if t.name not in complete_map.keys():
            import pdb; pdb.set_trace()

        opt_value = opt_value.rescale(complete_map[t.name])
        n, bins, patches = ax.hist(sorted(t.data), label=str(cell)+str(t.name))
        mode0 = bins[np.where(n==np.max(n))[0][0]]
        try:
            # mode = mode0*qt.unitless
            mode0 = mode0.rescale(opt_value)
            half = (bins[1]-bins[0])/2.0
            td = sorted(t.data)
            td = [t*qt.unitless for t in td]
            td = [t.rescale(opt_value) for t in td]
        except:
            import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        plt.hist(sorted(t.data), label=str(cell)+str(t.name))
        try:
            plt.scatter(opt_value,np.max(n),c='r',label='optima')

        except:
            plt.scatter(opt_value,np.max(n),c='r',label='optima')
        plt.savefig(str('optima_')+str(cell)+str(t.name)+str('.png'))
def data_versus_optimal(ga_out):
    rts,complete_map = pickle.load(open('../tests/russell_tests.p','rb'))

    dtcpop = [ p.dtc for p in ga_out['pf'] ]
    pop = [ p for p in ga_out['pf'] ]
    # first a nice violin plot of the test data.
    to_norm = np.matrix([list(t.data) for t in tests ])

    X,before = scale(to_norm)

    ax = sns.violinplot(x="test type", y="physical unit", hue="smoker",
                 data=X, palette="muted")

    for t in tests:
        plt.clf()
        fig, ax = plt.subplots()
        if t.name not in ga_out['pf'][0].dtc.prediction.keys():
            try:
                pred = ga_out['pf'][0].dtc.prediction['RheobaseTestP']

            except:
                pred = ga_out['pf'][0].dtc.rheobase
            if not isinstance(pred, dict):
                pred = {'value':pred}
        else:
            pred = ga_out['pf'][0].dtc.prediction[t.name]
        try:
            opt_value = pred['value']
        except:
            opt_value = pred['mean']
        if t.name not in complete_map.keys():
            import pdb; pdb.set_trace()

        opt_value = opt_value.rescale(complete_map[t.name])
        n, bins, patches = ax.hist(sorted(t.data), label=str(cell)+str(t.name))
        mode0 = bins[np.where(n==np.max(n))[0][0]]
        try:
            # mode = mode0*qt.unitless
            mode0 = mode0.rescale(opt_value)
            half = (bins[1]-bins[0])/2.0
            td = sorted(t.data)
            td = [t*qt.unitless for t in td]
            td = [t.rescale(opt_value) for t in td]
        except:
            import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        plt.hist(sorted(t.data), label=str(cell)+str(t.name))
        try:
            plt.scatter(opt_value,np.max(n),c='r',label='optima')

        except:
            plt.scatter(opt_value,np.max(n),c='r',label='optima')
        plt.savefig(str('optima_')+str(cell)+str(t.name)+str('.png'))

def ridge_regression(X_train, Y_train, X_test, Y_test, model_alpha):
    clf = linear_model.Ridge(model_alpha)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    loss = np.sum((predictions - Y_test)**2)
    return loss

def lasso_regression(X_train, Y_train, X_test, Y_test, model_alpha):
    clf = linear_model.Lasso(model_alpha)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    loss = np.sum((predictions - Y_test)**2)
    return loss
# https://stackoverflow.com/questions/35714772/sklearn-lasso-regression-is-orders-of-magnitude-worse-than-ridge-regression

def stochastic_gradient_descent(ga_out):
    Y = [ np.sum(v.fitness.values) for k,v in ga_out['history'].genealogy_history.items() ]
    X = [ list(v.dtc.attrs.values()) for k,v in ga_out['history'].genealogy_history.items() ]
    #ordered_attrs = set(ind.dtc.attrs.keys() for ind in ga_out['history'].genealogy_history[1])
    ordered_attrs = list(ga_out['history'].genealogy_history[1].dtc.attrs.keys())

    le = preprocessing.LabelEncoder()
    le.fit(ordered_attrs)
    le.classes_

    X = np.matrix(X)
    X,before = scale(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    sgd = SGDClassifier(penalty='l2', max_iter=1000, learning_rate='constant' , eta0=0.001  )

    sgd = SGDRegressor(penalty='l2', max_iter=1000, learning_rate='constant' , eta0=0.001  )
    sklearn.neural_network.MLPRegressor
    dnn = MLPClassifier(hidden_layer_sizes=(len(X),len(Y) ), activation='relu',
        solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant',
        learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
        random_state=None, tol=0.0001, verbose=True, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False,
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        n_iter_no_change=10)
    sgd.fit(X_train, Y_train)
    dnn.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    sklearn_sgd_predictions = sgd.predict(X_test)

    losslasso = lasso_regression(X_train, Y_train, X_test, Y_test, 0.3)
    lossridge = ridge_regression(X_train, Y_train, X_test, Y_test, 0.3)

    print(sklearn_sgd_predictions)
    print(ga_out['pf'][0].dtc.attrs)
    delta_y = Y_test - sklearn_sgd_predictions
    #pl.matshow(cm)
    #pl.title('Confusion matrix of the classifier')
    #pl.colorbar()
    #pl.show()

    return sgd, losslasso, lossridge




def init_pop(pop, td, tests):
    from neuronunit.optimisation.exhaustive_search import update_dtc_grid
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
'''
Depricated
def new_single_gene(pop,dtcpop,td):
    # some times genes explored will not return
    # un-usable simulation parameters
    # genes who have no rheobase score
    # will be discarded.
    #
    # optimiser needs a stable
    # gene number however
    #
    # This method finds how many genes have
    # been discarded, and tries to build new genes
    # from the existing distribution of gene values, by mimicing a normal random distribution
    # of genes that are not deleted.
    # if a rheobase value cannot be found for a given set of dtc model more_attributes
    # delete that model, or rather, filter it out above, and make new genes based on
    # the statistics of remaining genes.
    # it's possible that they wont be good models either, so keep trying in that event.
    # a new model from the mean of the pre-existing model attributes.

    # TODO:
    # boot_new_genes is a more standard, and less customized implementation of the code below.
    # pop, dtcpop = boot_new_genes(number_genes,dtcpop,td)
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    # Whats more boot_new_genes maps many to many (efficient).
    # Code below is maps, many to one ( less efficient).
    impute_gene = [] # impute individual, not impute index
    ind = WSListIndividual()
    for t in td:
        #try:
        mean = np.mean([ d.attrs[t] for d in dtcpop ])
        std = np.std([ d.attrs[t] for d in dtcpop ])
        sample = numpy.random.normal(loc=mean, scale=2*np.abs(std), size=1)[0]
            #import pdb; pdb.set_trace()
        ind.append(sample)
    dtc = DataTC()
    # Brian and PyNN models should not have to read from a file.
    # This line satifies an older NU design flaw, that all models evaluated must have
    # a disk readable path.
    # LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
    dtc.attrs = {}
    for i,j in enumerate(ind):
        dtc.attrs[str(td[i])] = j
    dtc.backend = dtcpop[0].backend
    dtc.tests = dtcpop[0].tests
    dtc = dtc_to_rheo(dtc)
    #print(dtc.rheobase)
    ind.rheobase = dtc.rheobase
    return ind,dtc
'''

def filtered(pop,dtcpop):
    dtcpop = [ dtc for dtc in dtcpop if type(dtc.rheobase) is not type(None) ]
    pop = [ p for p in pop if type(p.rheobase) is not type(None) ]
    if len(pop) != len(dtcpop):
        import pdb; pdb.set_trace()

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
    #if not hasattr(thing,'keys'):
    #    return thing
    if 'value' in thing.keys():
        return 'value'
    if 'mean' in thing.keys():
        return 'mean'

def dtc2gene(pop,dtcpop):
    fitness_attr = pop[0].fitness
    if type(fitness_attr) is type(None):
        import pdb; pdb.set_trace()
    for i,p in enumerate(pop):
        if not hasattr(p,'fitness'):
            p.fitness = fitness_attr

    for i,ind in enumerate(pop):
        if not hasattr(ind,'error_length') or ind.error_length is None:
            pop[i].error_length = dtcpop[0].error_length
        if not hasattr(ind,'backend') or ind.backend is None:
            pop[i].backend = dtcpop[0].backend

        if not hasattr(ind,'boundary_dict') or ind.boundary_dict is None:
            pop[i].boundary_dict = dtcpop[0].boundary_dict
    return pop

'''
def dtc2model(pop,dtcpop):
    for i,ind in enumerate(pop):
        pop[i].rheobase = dtcpop[i].rheobase
    return pop
'''

def pop2dtc(pop,dtcpop):
    for i,dtc in enumerate(dtcpop):
        if not hasattr(dtc,'backend') or dtc.backend is None:
            dtcpop[i].backend = pop[0].backend
        if not hasattr(dtc,'boundary_dict') or dtc.boundary_dict is None:
            dtcpop[i].boundary_dict = pop[0].boundary_dict
    return dtcpop

'''
def resample_high_sampling_freq(dtcpop):
    for d in dtcpop:
        d.attrs['dt'] = d.attrs['dt'] *(1.0/21.41)

    (dtcpop,_) = add_dm_properties_to_cells(dtcpop)
    return dtcpop
'''
def score_attr(dtcpop,pop):
    for i,d in enumerate(dtcpop):
        if not hasattr(pop[i],'dtc'):
            pop[i].dtc = None
        d.get_ss()
        pop[i].dtc = copy.copy(d)
    return dtcpop,pop
def get_dm(pop,dtcpop,tests,td):
    if CONFIDENT:
        dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
        dtcpop = list(dtcbag.map(nuunit_dm_evaluation).compute())
    else:
        dtcpop = list(map(nuunit_dm_evaluation,dtcpop))
    dtcpop,pop = score_attr(dtcpop,pop)
    return dtcpop,pop

def eval_subtest(name):
    for dtc in dtcpop:
        for t in dtc.tests:
            if name in t.name:
                score, dtc = bridge_judge((t, dtc))


import pdb
import quantities as pq
from sciunit import scores

def bridge_passive(package):
    t,dtc = package
    #model = mint_generic_model(dtc.backend)

    # model = mint_generic_model(dtc.backend)
    # model.set_attrs(**dtc.attrs)
    # this is neater but might break parallelism
    #model = dtc.dtc_to_model()
    model = new_model(dtc)
    t.setup_protocol(model)
    result = t.get_result(model)

    assert 'mean' in t.observation.keys()
    try:
        pred = t.extract_features(model, result)
    except:
        pred = None
        return None,dtc
    if 'std' in pred.keys():
        if float(pred['std']) == 0.0:
            pred['std'] = 1.0 * pred['mean'].units

    else:
        if not 'mean' in pred.keys():
            pred['mean'] = pred['value']

    assert 'mean' in t.observation.keys()
    # if    'std' not in t.observation.keys():
    take_anything = list(t.observation.values())[0]
    if 'std' not in t.observation.keys():
        t.observation['std'] = 5 * take_anything.magnitude * take_anything.units
    take_anything = list(pred.values())[0]
    if 'std' not in pred.keys():
        pred['std'] = 5 * take_anything.magnitude * take_anything.units

    if not hasattr(dtc,'predictions'):
        dtc.predictions = {}
        dtc.predictions[t.name] = pred
    else:
        dtc.predictions[t.name] = pred
    if RATIO_SCORE:
        t.score_type = scores.RatioScore

        score = t.compute_score(t.observation, pred)
    else:
        print('Ratio score failed')
        t.score_type = scores.ZScore

        score = t.compute_score(t.observation, pred)

    model = new_model(dtc)
    t.setup_protocol(model)
    result = t.get_result(model)
    '''
    standard deviations erroneously set to 0, in imputate predictions
    make simulated samples
    '''
    assert 'mean' in t.observation.keys()
    pred = t.extract_features(model,result)
    if 'std' in pred.keys():
        if float(pred['std']) == 0.0:
            pred['std'] = 1.0*pred['mean'].units

    else:
        if not 'mean' in pred.keys():
            pred['mean'] = pred['value']

    assert 'mean' in t.observation.keys()
    #if 'std' not in t.observation.keys():
    take_anything = list(t.observation.values())[0]
    if 'std' not in t.observation.keys():
        std.observation['std'] = 15*take_anything.magnitude * take_anything.units
    take_anything = list(pred.values())[0]
    pred['std'] = 15*take_anything.magnitude * take_anything.units

    score = t.compute_score(t.observation,pred)
    return score, dtc

def can_this_map1(dtc):
    return dtc

def can_this_map(dtc):

    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    tests = dtc.tests
    if not hasattr(dtc,'scores') or dtc.scores is None:
        dtc.scores = None
        dtc.scores = {}

    if isinstance(dtc.rheobase,type(None)) or type(dtc.rheobase) is type(None):
        dtc = allocate_worst(tests, dtc)
        #if self.verbose:
        #    print('score worst via test failure at {0}'.format('rheobase'))
    else:

        for k, t in enumerate(tests):
            key = str(t)
            dtc.scores[key] = 1.0
            #dtc = self.format_test(dtc)
            t.params = dtc.protocols[k]
            if 'mean' not in t.observation.keys():

                t.observation['mean']  = t.observation['value']
                assert 'mean' in t.observation.keys()
            #model = dtc.dtc_to_model()

            #model = mint_generic_model(dtc.backend)
            model = new_model(dtc)
            if not t.passive:
                pred = t.generate_prediction(model)
            else:
                pass
    return dtc
    '''
    t.setup_protocol(model)
    result = t.get_result(model)

    assert 'mean' in t.observation.keys()
    pred = t.extract_features(model, result)
    if 'std' in pred.keys():
        if float(pred['std']) == 0.0:
            pred['std'] = 1.0 * pred['mean'].units

    else:
        if not 'mean' in pred.keys():
            pred['mean'] = pred['value']

    assert 'mean' in t.observation.keys()
    # if    'std' not in t.observation.keys():
    take_anything = list(t.observation.values())[0]
    t.observation['std'] = 15 * take_anything.magnitude * take_anything.units
    take_anything = list(pred.values())[0]
    pred['std'] = 15 * take_anything.magnitude * take_anything.units
    return dtc
    '''


class OptMan():
    def __init__(self,tests, td=None, backend = None,hc = None,boundary_dict = None, error_length=None,protocol=None,simulated_obs=None,verbosity=None,confident=None,tsr=None):
        self.tests = tests
        self.td = td
        if tests is not None:
            self.error_length = len(tests)
        self.backend = backend
        self.hc = hc
        self.boundary_dict= boundary_dict
        self.protocol = protocol
        if type(confident) is type(None):
            self.CONFIDENT = True
        else:
            self.CONFIDENT = confident
        if verbosity is None:
            self.verbose = 0
        else:
            self.verbose = verbosity
        if type(tsr) is not None:
            self.tsr = tsr

    def new_single_gene(self,dtc,td):
        from neuronunit.optimisation.optimisations import SciUnitOptimisation

        import random
        from datetime import datetime
        random.seed(datetime.now())
        #random.seed(64)
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
        #pop = []
        gene = DO.set_pop(boot_new_random=1)
        #import pdb
        #pdb.set_trace()
        #gene = dtc2model(gene,dtc)
        dtc_ = self.update_dtc_pop(gene,self.td)
        dtc_ = pop2dtc(gene,dtc_)
        return gene[0], dtc_[0]



    def get_allen(self,pop,dtcpop,tests,td,tsr=None):
        with open('waves.p','rb') as f:
            make_stim_waves = pickle.load(f)
            #import pdb; pdb.set_trace()
        NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])
        for dtc in dtcpop: dtc.spike_number = tests['spike_count']['mean']
        for dtc in dtcpop: dtc.pre_obs = None
        for dtc in dtcpop: dtc.pre_obs = self.tests
        for dtc in dtcpop: dtc.tsr = tsr #not a property but an aim
        #        assert
        if CONFIDENT==True:
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

        if CONFIDENT == True:
            dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
            dtcpop = list(dtcbag.map(nuunit_allen_evaluation).compute())

        else:
            dtcpop = list(map(nuunit_allen_evaluation,dtcpop))

        return pop, dtcpop



    def round_trip_test(self,tests,backend,free_paramaters=None,NGEN=None,MU=None,mini_tests=None,stds=None):
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
                    #self.simulated_obs = True
                    new_tests = TSD(new_tests)
                    new_tests.use_rheobase_score = tests.use_rheobase_score

                    ga_out, DO = run_ga(ranges,NGEN,mt,free_params=rp.keys(), MU = MU, backend=backend, selection=str('selNSGA2'),protocol={'elephant':True,'allen':False})
                    results[k] = copy.copy(ga_out['pf'][0].dtc.scores)

            else:
                temp = new_tests['RheobaseTest'].observation
                if type(temp) is not type({'0':1}):
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

                ga_out, DO = run_ga(ranges,NGEN,new_tests,free_params=chosen_keys, MU = MU, backend=backend, selection=str('selNSGA2'))
                results = copy.copy(ga_out['pf'][0].dtc.scores)
            ga_converged = [ p.dtc for p in ga_out['pf'][0:2] ]
            test_origin_target = [ dsolution for i in range(0,len(ga_out['pf'])) ][0:2]
            if self.verbose:
                print(ga_out['pf'][0].dtc.scores)
                print(results)
                print(ga_out['pf'][0].dtc.attrs)

            inject_and_plot(ga_converged,second_pop=test_origin_target,third_pop=[ga_converged[0]],figname='not_a_problem.png',snippets=True)
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

    def round_trip_test_rheob(self,tests,backend,free_paramaters=None,NGEN=None,MU=None):
        '''
        # Inputs:
        #    -- tests, a list of NU test types,
        #    -- backend a string encoding what model, backend, simulator to use.
        # Outputs:
        #    -- a score, that should be close to zero larger is worse.
        # Synopsis:
        #    -- Given any models
        # lets check if the optimiser can find arbitarily sampeled points in
        # a parameter space, using only the information in the error gradient.
        # make some new tests based on internally generated data
        # as opposed to experimental data.
        '''

        out_tests = []

        #dsolution.rheobase = tests['RheobaseTest'].observation['value']
        if NGEN is None:
            NGEN = 10
        if MU is None:
            MU = 10
        ranges = MODEL_PARAMS[backend]

        if self.protocol['allen']:
            dtc = False
            while dtc is False:
                dsolution,rp,chosen_keys,random_param = process_rparam(backend)
                free_params = random_param.keys()

                dtc = self.make_simulated_observations(tests,backend,rp)
                dsolution.tests = tests
            #dtc = new_tests
            observations = dtc.preds
            target_spikes = dtc.spike_number+10
            observation_spike = {'value': target_spikes}
            ga_out, DO = run_ga(ranges,NGEN,observations,free_params=free_params, MU = MU, backend=backend, selection=str('selNSGA3'), protocol={'allen':True,'elephant':False})
            ga_out.b = None
            ga_out.bdtc = ga_out['pf'][0].dtc
            dtcpop0 = [ p.dtc for p in ga_out['pf'] ]
            dtcpop1 = [ dsolution for i in range(0,len(ga_out['pf'])) ]
            inject_and_plot(dtcpop0,second_pop=dtcpop1,third_pop=[dtcpop0[0],dtcpop0[-1]],figname='snippets.png',snippets=True)

        elif self.protocol['elephant']:
            new_tests = False
            while new_tests is False:
                dsolution,rp,chosen_keys,random_param = process_rparam(backend)
                free_params = random_param.keys()

                new_tests = self.make_simulated_observations(tests,backend,rp)

            if free_paramaters is None:
                fp = chosen_keys
            else:
                fp = free_paramaters
            dsolution.rheobase = new_tests['RheobaseTest'].observation
            import pdb
            mini_tests = {}
            cnt=0
            while check_test(new_tests):
                new_tests = self.make_simulated_observations(tests,backend,rp)
                if self.verbose:
                    print('in loop cnt=: {0}'.format(cnt))
            results = {}
            mini_tests = {'RheobaseTest': new_tests['RheobaseTest']}

            ga_out, DO = run_ga(ranges,NGEN,mini_tests,free_params=fp, MU = MU, backend=backend, selection=str('selNSGA2'))

            #for t in new_tests:
            dtcpop0 = [ p.dtc for p in ga_out['pf'][0:2] ]
            dtcpop1 = [ dsolution for i in range(0,len(ga_out['pf'])) ][0:2]
            if self.verbose:
                print(ga_out['pf'][0].dtc.scores)
                print(results)

                print(ga_out['pf'][0].dtc.attrs)

            return ga_out,dtcpop0,dtcpop1


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
        else:
            dtc.model_path = None
            dtc.model_path = path_params['model_path']
        dtc.preds = None
        dtc.preds = {}
        dtc = dtc_to_rheo(dtc)

        dtc = self.format_test(dtc)
        tests = dtc.tests

        for k,t in enumerate(tests):
            #if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
            t.params = dtc.protocols[k]

            test_and_models = (t, dtc)
            pred = pred_only(test_and_models)
            dtc.preds[str(t.name)] = pred

        #else:
        #        dtc.preds[str(t.name)] = dtc.rheobase
        return dtc

    def elephant_evaluation(self,dtc):
        # Inputs single data transport container modules, and neuroelectro observations that
        # inform test error error_criterion
        # Outputs Neuron Unit evaluation scores over error criterion
        tests = dtc.tests
        if not hasattr(dtc,'scores') or dtc.scores is None:
            dtc.scores = None
            dtc.scores = {}

        if isinstance(dtc.rheobase,type(None)) or type(dtc.rheobase) is type(None):
            dtc = allocate_worst(tests, dtc)
            if self.verbose:
                print('score worst via test failure at {0}'.format('rheobase'))
        else:

            for k, t in enumerate(tests):
                try:
                    print(self.tests.use_rheobase_score)
                except:
                    import pdb
                    pdb.set_trace()
                if self.tests.use_rheobase_score == False and "RheobaseTest" in str(k):
                    continue
                key = str(t)
                dtc.scores[key] = 1.0
                #dtc = self.format_test(dtc)
                t.params = dtc.protocols[k]
                if 'mean' not in t.observation.keys():

                    t.observation['mean']  = t.observation['value']
                    assert 'mean' in t.observation.keys()
                if t.passive is False:
                    #model = dtc.dtc_to_model()
                    model = new_model(dtc)
                    pred = t.generate_prediction(model)
                    take_anything = list(t.observation.values())[0]
                    if 'std' not in t.observation.keys():
                        t.observation['std'] = 15*take_anything.magnitude * take_anything.units

                    take_anything = list(pred.values())[0]
                    if take_anything is None:
                        continue
                    pred['std'] = 15*take_anything.magnitude * take_anything.units

                    score, dtc = bridge_judge((t, dtc))
                else:
                    score, dtc = bridge_passive((t, dtc))


                assignment = 1.0
                #import pdb
                if score is not None:
                    if score.norm_score is not None:
                        assignment = 1.0 - score.norm_score
                dtc.scores[key] = assignment
        dtc.summed = dtc.get_ss()
        try:
            greatest = np.max([dtc.error_length,len(dtc.scores)])
        except:
            greatest = len(dtc.scores)
        dtc.scores_ratio = dtc.summed/greatest
        return dtc


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
    def format_test(self,dtc):
        # pre format the current injection dictionary based on pre computed
        # rheobase values of current injection.
        # This is much like the hooked method from the old get neab file.
        dtc.protocols = {}
        if not hasattr(dtc,'tests'):
            dtc.tests = copy.copy(self.tests)

        if hasattr(dtc.tests,'keys'):# is type(dict):
            tests = [key for key in dtc.tests.values()]
            dtc.tests = switch_logic(tests)#,self.tests.use_rheobase_score)
        else:
            dtc.tests = switch_logic(dtc.tests)
        for k,v in enumerate(dtc.tests):
            dtc.protocols[k] = {}
            if hasattr(v,'passive'):#['protocol']:
                if v.passive == False and v.active == True:
                    keyed = dtc.protocols[k]#.params
                    dtc.protocols[k] = active_values(keyed,dtc.rheobase)

                elif v.passive == True and v.active == False:
                    keyed = dtc.protocols[k]#.params
                    dtc.protocols[k] = passive_values(keyed)
            if v.name in str('RestingPotentialTest'):

                dtc.protocols[k]['injected_square_current']['amplitude'] = 0.0*pq.pA
        return dtc

    def make_simulated_observations(self,xtests,backend,random_param,dsolution=None):
        #self.simulated_obs = True
        # to be used in conjunction with round_trip_test below.

        if dsolution is None:
            dtc = DataTC()
            dtc.attrs = random_param
            dtc.backend = copy.copy(backend)
        else:
            dtc = dsolution
            #dsolution = dtc
        if self.protocol['elephant']:
            if str('RheobaseTest') in xtests.keys():
                dtc = get_rh(dtc,xtests['RheobaseTest'])
                if type(dtc.rheobase) is type(float(0.0)):
                    pass
                if type(dtc.rheobase) is type({'1':0}):
                    if dtc.rheobase['value'] is None:
                        return False, dtc
            dtc = make_new_random(dtc, copy.copy(backend))
            xtests = list(xtests.values())
            dtc.tests = xtests
            simulated_observations = {t.name:copy.copy(t.observation['value']) for t in xtests}
            simulated_observations = {k:v for k,v in simulated_observations.items() if v is not None}
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
                    simulated_observations['RheobaseTest']['value'] = temp
                    break
                else:
                    continue

            simulated_observations = {k:p for k,p in simulated_observations.items() if type(k) is not type(None) and type(p) is not type(None) }
            simulated_tests = {}
            for k in xtests:
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

    def update_dtc_pop(self,pop, td):
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
            xargs = zip(pop,repeat(td),repeat(_backend))
            npart = np.min([multiprocessing.cpu_count(),len(pop)])
            bag = db.from_sequence(xargs, npartitions = npart)
            dtcpop = list(bag.map(transform).compute())

            assert len(dtcpop) == len(pop)
            for dtc in dtcpop:
                dtc.backend = self.backend
                dtc.boundary_dict = None
                dtc.boundary_dict = self.boundary_dict
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
            dtc.boundary_dict = None
            dtc.boundary_dict = self.boundary_dict
            return dtc


    def init_pop(self,pop, td, tests):
        #from neuronunit.optimisation.exhaustive_search import update_dtc_grid
        dtcpop = list(self.update_dtc_pop(pop, td))
        for d in dtcpop:
            d.tests = tests
            if self.backend is not None:
                d.backend = self.backend

        if self.hc is not None:
            constant = self.hc
            for d in dtcpop:
                if constant is not None:
                    if len(constant):
                        d.constants = constant
                        d.add_constant()

        return pop, dtcpop

    def obtain_rheobase(self,pop, td, tests):
        '''
        Calculate rheobase for a given population pop
        Ordered parameter dictionary td
        and rheobase test rt
        '''
        pop, dtcpop = self.init_pop(pop, td, tests)
        #for d in dtcpop:
        #    d.tests = tests
        if 'RAW' in self.backend  or 'HH' in self.backend or str('ADEXP') in self.backend:
            dtcpop = list(map(dtc_to_rheo,dtcpop))
            dtcpop = list(map(self.format_test,dtcpop))
        else:
            dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
            dtcpop = list(dtcbag.map(dtc_to_rheo))

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

    def parallel_route(self,pop,dtcpop,tests,td):
        NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])

        if self.protocol['allen']:
            pop, dtcpop = self.get_allen(pop,dtcpop,tests,td,tsr=self.protocol['tsr'])
            pop = [pop[i] for i,d in enumerate(dtcpop) if type(d) is not type(None)]
            dtcpop = [d for d in dtcpop if type(d) is not type(None)]
            return pop, dtcpop

        elif str('dm') in self.protocol.keys():
            if self.protocol['dm']:
                pdb.set_trace()
                pop, dtcpop = get_dm(pop,dtcpop,tests,td)
                return pop, dtcpop

        elif self.protocol['elephant']:
            for d in dtcpop:
                d.tests = copy.copy(self.tests)


            if CONFIDENT == True:# and self.backend is not str('ADEXP'):
                dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
                dtcpop = list(dtcbag.map(self.format_test).compute())

                for d in dtcpop:
                    assert hasattr(d, 'tests')

                if str('ADEXP') in self.backend:
                    serial_dtc = list(map(self.elephant_evaluation,dtcpop))
                    # A way to get parallelism backinto brian2.
                    serial_dtc = self.serial_dtc(dtcpop,b=True)
                    parallel_dtc = self.serial_dtc(dtcpop,b=False)
                    dtcbag = db.from_sequence(parallel_dtc, npartitions = NPART)
                    parallel_dtc = list(dtcbag.map(self.elephant_evaluation).compute())
                    serial_dtc = list(map(self.elephant_evaluation,serial_dtc))
                    dtcpop = []
                    for i,j in zip(serial_dtc,parallel_dtc):
                        i.tests.extend(j.tests)
                        i.tests.extend(j.tests)
                        i.scores.update(j.scores)
                        dtcpop.append(i)
                else:
                    dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
                    dtcpop = list(dtcbag.map(self.elephant_evaluation).compute())


                for d in dtcpop:
                    assert hasattr(d, 'tests')

                for d in dtcpop:
                    d.tests = copy.copy(self.tests)

            else:
                dtcpop = list(map(self.format_test,dtcpop))
                dtcpop = list(map(self.elephant_evaluation,dtcpop))
                for d in dtcpop:
                    d.tests = copy.copy(self.tests)


                #def skip_over(dtcpop):
                #    # hint at how to simplify above
                #    dtcpop = [dtc.dtc_to_predictions() for dtc in dtcpop]#list(dtcbag.map(dtc_to_predictions).compute())

            for d in dtcpop:
               if not hasattr(d, 'tests'):
                 import pdb
                 pdb.set_trace()
        return pop, dtcpop

    def test_runner(self,pop,td,tests):
        if self.protocol['elephant']:
            pop_, dtcpop = self.obtain_rheobase(pop, td, tests)
            for ind,dtc in zip(pop,dtcpop):
                dtc.error_length = self.error_length
            pop, dtcpop = self.make_up_lost(copy.copy(pop_), dtcpop, td)

            # there are many models, which have no actual rheobase current injection value.
            # filter, filters out such models,
            # gew genes, add genes to make up for missing values.
            # delta is the number of genes to replace.

        elif self.protocol['allen']:

            pop, dtcpop = self.init_pop(pop, td, tests)
            for ind,dtc in zip(pop,dtcpop):
                dtc.error_length = self.error_length

            for ind,d in zip(pop,dtcpop):
                d.error_length = self.error_length
                ind.error_length = self.error_length
        pop,dtcpop = self.parallel_route(pop, dtcpop, tests, td)#, clustered=False)
        both = [(ind,dtc) for ind,dtc in zip(pop,dtcpop) if dtc.scores is not None]
        for ind,d in both:
            ind.dtc = None
            ind.dtc = d
            if d.scores is not None:
                ind = copy.copy(both[0][0])
                d = copy.copy(both[0][1])

            if not hasattr(ind,'fitness'):
                ind.fitness = copy.copy(pop_[0].fitness)
                for i,v in enumerate(list(ind.fitness.values)):
                    ind.fitness.values[i] = list(ind.dtc.evaluate.values())[i]
        pop = [ ind for ind,d in zip(pop,dtcpop) if d.scores is not None ]
        dtcpop = [ d for ind,d in zip(pop,dtcpop) if d.scores is not None ]
        return pop,dtcpop

    def make_up_lost(self,pop,dtcpop,td):
        '''
        make new genes.
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
                    #assert len(self.tests)>1
                    dtc = self.format_test(dtc)
                    ind.dtc = dtc
                #import pdb
                #pdb.set_trace()
                pop_ = [ p for p in pop_ if len(p)>1 ]
                #if cnt>=2 or not len(pop_):
                #pop.extend(pop[0])
                #dtcpop.extend(dtcpop[0])
                #else:
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

    def update_deap_pop(self,pop, tests, td, backend = None,hc = None,boundary_dict = None, error_length=None):
        '''
        Inputs a population of genes (pop).
        Returned neuronunit scored DataTransportContainers (dtcpop).
        This method converts a population of genes to a population of Data Transport Containers,
        Which act as communicatable data types for storing model attributes.
        Rheobase values are found on the DTCs
        DTCs for which a rheobase value of x (pA)<=0 are filtered out
        DTCs are then scored by neuronunit, using neuronunit models that act in place.
        '''
        if len(pop)==0:
            raise Exception('User error population size set to 0')
        if hc is not None:
            self.td = None
            self.td = self.td

        if hc is not None:
            self.hc = None
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
        return pop
    def boot_new_genes(self,number_genes,dtcpop,td):
        '''
        Boot strap new genes to make up for completely called onesself.
        '''
        from neuronunit.optimisation.optimisations import SciUnitOptimisation
        import random
        from datetime import datetime
        random.seed(datetime.now())
        DO = SciUnitOptimisation(offspring_size = number_genes,
                                 error_criterion = self.tests, boundary_dict = dtcpop[0].boundary_dict,
                                 backend = dtcpop[0].backend, selection = str('selNSGA'),protocol = self.protocol)#,, boundary_dict = ss, elite_size = 2, hc=hc)
        DO.setnparams(nparams = len(dtcpop[0].attrs), boundary_dict = dtcpop[0].boundary_dict)
        DO.setup_deap()
        # pop = []
        if 1==number_genes:
            pop = DO.set_pop(boot_new_random=5)
        if 1<number_genes and number_genes<5:
            pop = DO.set_pop(boot_new_random=5)

        else:
            pop = DO.set_pop(boot_new_random=number_genes)
        pop = dtc2gene(pop,dtcpop)
        dtcpop_ = self.update_dtc_pop(pop,td)
        dtcpop_ = pop2dtc(pop,dtcpop_)
        dtcpop_ = list(map(dtc_to_rheo,dtcpop_))
        for i,ind in enumerate(pop):
            pop[i].rheobase = dtcpop_[i].rheobase
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
