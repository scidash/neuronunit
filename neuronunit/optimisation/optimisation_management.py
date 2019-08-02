#import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
import matplotlib
matplotlib.use('agg')

#    Goal is based on this. Don't optimize to a singular point, optimize onto a cluster.
#    Golowasch, J., Goldman, M., Abbott, L.F, and Marder, E. (2002)
#    Failure of averaging in the construction
#    of conductance-based neuron models. J. Neurophysiol., 87: 11291131.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import dask.bag as db
import pandas as pd
import dask.bag as db
import os
import pickle
from collections import Iterable, OrderedDict
# The rheobase has been obtained seperately and cannot be db mapped.
# Nested DB mappings dont work.
from itertools import repeat
import neuronunit
import multiprocessing
npartitions = multiprocessing.cpu_count()
#from collections import Iterable
from numba import jit
from sklearn.model_selection import ParameterGrid
from itertools import repeat
from collections import OrderedDict


import logging
logger = logging.getLogger('__main__')
logger.debug('test')
import copy
import math
import quantities as pq
import numpy as np
from itertools import repeat
import numpy
from sklearn.cluster import KMeans

from deap import base
from pyneuroml import pynml
import dask.array as da

from neuronunit.optimisation.data_transport_container import DataTC
#from neuronunit.models.interfaces import glif

# Import get_neab has to happen exactly here. It has to be called only on
from neuronunit import tests
from neuronunit.optimisation import get_neab
#from neuronunit.models.reduced import ReducedModel
from neuronunit.optimisation.model_parameters import path_params
from neuronunit.optimisation import model_parameters as modelp


from neuronunit.tests.fi import RheobaseTestP# as discovery
from neuronunit.tests.fi import RheobaseTest# as discovery
from neuronunit.tests.druckman2013 import *
#from neuronunit.tests.base import PASSIVE_DURATION, PASSIVE_DELAY

import dask.bag as db
# The rheobase has been obtained seperately and cannot be db mapped.
# Nested DB mappings dont work.
from itertools import repeat
import efel


#DURATION = 1000.0*pq.ms
#DELAY = 100.0*pq.ms
from neuronunit.tests.base import AMPL, DELAY, DURATION
try:
    from neuronunit.tests.base import passive_AMPL, passive_DELAY, passive_DURATION
except:
    pass
import efel

from allensdk.ephys.extract_cell_features import extract_cell_features

from sciunit.models.runnable import RunnableModel
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from collections.abc import Iterable

#DURATION = 2000
#DELAY = 200

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


def inject_rh_and_dont_plot(dtc):
    '''
    For debugging backends during development.
    '''
    model = mint_generic_model(dtc.backend)

    try:
        rheobase = dtc.rheobase['value']
    except:
        rheobase = dtc.rheobase
    #import pdb
    #pdb.set_trace()
    uc = { 'injected_square_current': {'amplitude':rheobase,'duration':DURATION,'delay':DELAY }}

    dtc.run_number += 1
    model.set_attrs(**dtc.attrs)

    model.inject_square_current(uc['injected_square_current'])
    return (model, model.get_membrane_potential().times,model.get_membrane_potential(),uc)

def inject_and_plot(dtc,second_pop=None,figname='problem'):

    import seaborn as sns

    if not isinstance(dtc, Iterable):
        model = mint_generic_model(dtc.backend)
        try:
            rheobase = dtc.rheobase['value']
        except:
            rheobase = dtc.rheobase

        uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
        dtc.run_number += 1
        model.set_attrs(**dtc.attrs)
        model.inject_square_current(uc)
        if str(dtc.backend) is str('ADEXP'):
            model.finalize()
        else:
            print(str(dtc.backend))
        #vm = model.get_membrane_potential().magnitude
        sns.set_style("darkgrid")
        plt.plot(model.get_membrane_potential().times,vm.magnitude)#,label='ground truth')
        plot_backend = mpl.get_backend()
        if plot_backend == str('Agg'):
            plt.savefig(figname+str('debug.png'))
        else:
            plt.show()

    else:
        if type(second_pop) is not type(None):
            dtcpop = copy.copy(dtc)
            dtc = None

            fig = plt.figure(figsize=(11,8.5))
            ax = fig.add_subplot(111)
            for dtc in dtcpop:
                model = mint_generic_model(dtc.backend)
                try:
                    rheobase = dtc.rheobase['value']
                except:
                    rheobase = dtc.rheobase
                uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    print(str(dtc.backend))
                vm = model.get_membrane_potential().magnitude
                print(np.max(vm))
                sns.set_style("darkgrid")
                plt.plot(model.get_membrane_potential().times,vm,label=dtc.backend)#,label='ground truth')

            for dtc in second_pop:
                model = mint_generic_model(dtc.backend)
                try:
                    rheobase = dtc.rheobase['value']
                except:
                    rheobase = dtc.rheobase
                uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    print(str(dtc.backend))
                vm = model.get_membrane_potential().magnitude
                sns.set_style("darkgrid")
                plt.plot(model.get_membrane_potential().times,vm,label=dtc.backend)#,label='ground truth')
                #ax.legend(['A simple line'])
                ax.legend()
            plot_backend = mpl.get_backend()
            if plot_backend == str('Agg'):
                plt.savefig(figname+str('all_traces.png'))
            else:
                plt.show()
        else:
            dtcpop = copy.copy(dtc)
            dtc = None
            for dtc in dtcpop[0:2]:
                model = mint_generic_model(dtc.backend)
                try:
                    rheobase = dtc.rheobase['value']
                except:
                    rheobase = dtc.rheobase
                uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                vm = model.get_membrane_potential().magnitude
                sns.set_style("darkgrid")
                plt.plot(model.get_membrane_potential().times,vm)#,label='ground truth')
            plot_backend = mpl.get_backend()
            if plot_backend == str('Agg'):
                plt.savefig(figname+str('all_traces.png'))
            else:
                plt.show()
    return (model.get_membrane_potential().times,model.get_membrane_potential())



def cell_to_efel_mapper(content):
    dm_properties = {}
    index,dtc = content
    dm_properties[index] = []
    ir_currents = dtc.rheobase
    standard = 1.5*ir_currents
    strong = 3*ir_currents
    model = None
    model = mint_generic_model(dtc.backend)

    place_holder = {}
    place_holder['n'] = 86
    place_holder['mean'] = 10*pq.pA
    place_holder['std'] = 10*pq.pA
    place_holder['value'] = 10*pq.pA
    backend_ = dtc.backend
    #keyed['injected_square_current'] = {}

    if 'RAW' in backend_ or 'HH' in backend_ or 'GLIF' in backend_:#Backend:
        rtest = RheobaseTest(observation=place_holder,
                                name='a Rheobase test')
    else:
        rtest = RheobaseTestP(observation=place_holder,
                                name='a Rheobase test')
    score = rtest.judge(model)
    results = model.get_membrane_potential()
    trace = {}


    trace['T'] = results['vm'].times()
    trace['V'] = results['vm']
    #trace['stim_start'] = rtest.run_params[]
    trace['stim_end'] = list(w['duration'])
    traces = [trace1]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
    #traces_results = efel.getFeatureValues(traces,['AP_amplitude', 'voltage_base'])#
    return dtc


def cell_to_test_mapper(dtc):

    dm_properties = {}

    dm_properties['dm'] = []
    dm_properties['efel'] = []

    ir_currents = dtc.rheobase
    standard = 1.5*ir_currents
    strong = 3*ir_currents
    (_,times,vm,protocol) = inject_rh_and_dont_plot(dtc)
    dm_tests = init_dm_tests(standard,strong,params=protocol)
    model = mint_generic_model(dtc.backend)
    model.set_attrs(**dtc.attrs)

    max_vm = float(np.max(vm))
    ARTIFICIAL = False

    if max_vm < 0.0:
        ARTIFICIAL = True
        off_set = np.abs(0.0 - max_vm) + float(np.abs(np.mean(vm)))
        off_set = off_set*pq.mV
        vm = [ v+off_set for v in vm ]
        max_vm = float(np.max(vm))

    if not ARTIFICIAL:
        for i, test in enumerate(dm_tests):
            print(model.attrs, 'this is not none')
            dm_properties['dm'].append(test.generate_prediction(model)['mean'])
            print(dm_properties['dm'][-1])
        dtc.dm_properties = None
        dtc.dm_properties = dm_properties['dm']
    if max_vm > 0.0:
        import pickle
        pickle.dump([(model,times,vm)],open('efel_practice.p','wb'))
        (model,times,vm) = inject_rh_and_dont_plot(dtc)
        import pdb; pdb.set_trace()
        from neuronunit.capabilities.spike_functions import get_spike_waveforms
        waveforms = get_spike_waveforms(vm)

        trace = {}
        trace['T'] = waveforms[:,0].times
        trace['V'] = waveforms[:,0]

        trace['T'] = [ float(t) for t in trace['T'] ]
        trace['V'] = [ float(v) for v in trace['V'] ]
        trace['stim_start'] = [ 0 ] #[sm.complete['Time_Start']]#rtest.run_params[]
        trace['stim_end'] = [ 0 + float(np.max(trace['T'])) ]# list(sm.complete['duration'])
        traces = [trace]# Now we pass 'traces' to the efel and ask it to calculate the feature# values

        import pdb; pdb.set_trace()
        traces_results = efel.getFeatureValues(traces,list(efel.getFeatureNames()))#
        print(trace_results)
        dm_properties['efel'].append(traces_results)
        dtc.efel_properties = None
        dtc.efel_properties = dm_properties['efel']
    return dtc

confident = False
def add_dm_properties_to_cells(dtcpop,backend=str('static')):
    if len(dtcpop) > 8 and confident:
        bag = db.from_sequence(dtcpop,npartitions=8)
        dtcpop = list(bag.map(cell_to_test_mapper).compute())
    else:
        dtcpop = list(map(cell_to_test_mapper,dtcpop))
    dm_properties = {}

    for dtc in dtcpop:
        if hasattr(dtc,'dm_properties'):
            dm_properties.update(dtc.dm_properties)
    return (dtcpop,dm_properties)


def make_imputed_observations(tests,backend,random_param):
    '''
    to be used in conjunction with round_trip_test below.

    '''
    dtc = DataTC()
    dtc.attrs = random_param
    dtc.backend = backend
    dtc.tests  = tests
    dtc = get_rh(dtc,tests[0])
    dtc = pred_evaluation(dtc)

    # Temporarily patch sciunit judge code, which seems to be broken.
    #predictions = pred_only(test_and_models)
    #predictions = list(ptbag.map(obtain_predictions).compute())
    predictions.insert(0,dtc.rheobase)
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
    # lets check if the optimiser can find arbitarily sampeled points in
    # a parameter space, using only the information in the error gradient.
    # make some new tests based on internally generated data
    # as opposed to experimental data.
    '''
    ranges = MODEL_PARAMS[str(backend)]
    random_param = {} # randomly sample a point in the viable parameter space.
    for k in ranges.keys():
        try:
            mean = np.mean(ranges[k])
            std = np.std(ranges[k])
            sample = numpy.random.normal(loc=mean, scale=2*std, size=1)[0]
            random_param[k] = sample
        except:
            random_param[k] = ranges[k]
    tests = make_imputed_observations(tests,backend,random_param)
    NGEN = 10
    MU = 6
    ga_out, DO = run_ga(ranges,NGEN,tests,free_params=free_params, NSGA = True, MU = MU, backed=backend, selection=str('selNSGA2'))
    best = ga_out['pf'][0].dtc.get_ss()
    #print(Bool(best < 0.5))
    if Bool(best >= 0.5):
        NGEN = 10
        MU = 6
        ga_out, DO = run_ga(explore_param,NGEN,tests,free_params=free_params, NSGA = True, MU = MU, backed=backend, selection=str('selNSGA2'),seed_pop=pf[0].dtc.attrs)
        best = ga_out['pf'][0].dtc.get_ss()
    print('Its ',Bool(best < 0.5), ' that optimisation succeeds on this model class')
    print('goodness of fit: ',best)
    dtcpop = [ p.dtc for p in ga_out['pf'] ]
    return ( Bool(ga_out['pf'][0] < 0.5),dtcpop )


def pred_only(test_and_models):
    # Temporarily patch sciunit judge code, which seems to be broken.
    (test, dtc) = test_and_models
    obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    try:
        pred = test.generate_prediction(model)
    except:
        pred = None

    return pred

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
            dtc.prediction = {}
            dtc.prediction[test.name] = pred
            dtc.observation = {}
            dtc.observation[test.name] = test.observation['mean']


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

def get_centres(use_test,backend,explore_param):
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


def save_models_for_justas(dtc):
    with open(str(dtc.attrs)+'.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)

'''
def cluster_tests(use_test,backend,explore_param):
    Given a group of conflicting NU tests, quickly exploit optimisation, and variance information
    To find subsets of tests that don't conflict.
    Inputs:
        backend string, signifying model backend type
        explore_param list of dictionaries, of model parameter limits/ranges.
        use_test, a list of tests per experimental entity.
    Outputs:
        lists of groups of less conflicted test classes.
        lists of groups of less conflicted test class names.
    td, test_opt, centres = get_centres(use_test,backend,explore_param)
    cell_attrs = [ centers[:, 0],centers[:, 1], centers[:, 2] ]
    for key, use_test in test_frame.items():
        preds = []
        mean_scores = []

        Create a model situation analogous to the NeuroElectro data situation.
        Assume, that I have three or more clustered experimental observations,
        averaging wave measurements is inappropriate, but thats what I have.
        Try to reconstruct the clustered means, by clustering solution sets with respect to 8 waveform measurements.
        The key is to realize, that averaging measurements, and computing error is very different to, taking measurements, and averaging error.
        The later is the multiobjective approach to optimisation. The former is the approach used here.
        dtcpop = []
        for ca in cell_attrs:
            What is the mean waveform measurement resulting from averaging over the three data points?
            dtc = update_dtc_pop(ca, td)
            dtcpop.append(dtc)
            dtc.tests = use_test # aliased variable.
            if 'RAW' in dtc.backend  or 'HH' in dtc.backend :#Backend:
                #rtest = [ t for t in dtc.tests if str('RheobaseTest') == t.name ]
                dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
                dtcpop = list(dtcbag.map(dtc_to_rheo))
                dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
                dtcpop = list(dtcbag.map(format_test))
            else:
                dtcpop = map(dtc_to_rheo,dtcpop)
                dtcpop = map(format_test,dtcpop)
            test_and_dtc = (use_test,dtc)

            preds.append(pred_only(test_and_dtc))
        # waveform measurement resulting from averaging over the three data points?
        mean_pred = np.mean(preds)
        mean_score,dtc = score_only(dtc,mean_pred,test)
        mean_scores.append(mean_score)

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
    return (grouped_tests, grouped_tests, mean_scores)
'''

def mint_generic_model(backend):
    #LEMS_MODEL_PATH = path_params['model_path']
    #return ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = str(backend))
    #model = ReducedModel(dtc.model_path,name='vanilla', backend=(dtc.backend, {'DTC':dtc}))

    model = RunnableModel(str(backend),backend=backend)#, {'DTC':dtc}))

    return model


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
    model.set_attrs(dtc.attrs)
    pred = test.generate_prediction(model)
    return pred#(pred,obs)


def bridge_judge(test_and_dtc):
    # Temporarily patch sciunit judge code, which seems to be broken.
    #
    #
    (test, dtc) = test_and_dtc
    obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(**dtc.attrs)
    # pred = test.generate_prediction(model)
    #import pdb; pdb.set_trace()
    try:
        pred = test.generate_prediction(model)
    except:
        pred = None

    if type(pred) is not type(None):
        score = test.compute_score(test.observation,pred)
        if str('Rheobase') in str(test.name) and str(dtc.backend) in str('GLIF') :
            pass
            # consider not scoring rheobase
            #import pdb; pdb.set_trace()
            #score = 0.0
            #if model.get_spike_count() > 1:
            #    pass
                #print(model.get_spike_count)
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
    backend_ = dtc.backend
    #keyed['injected_square_current'] = {}

    if 'RAW' in backend_ or 'HH' in backend_ or 'ADEXP' in backend_ or 'GLIF' in backend_:#Backend:
        rtest = RheobaseTest(observation=place_holder,
                                name='a Rheobase test')
    else:
        rtest = RheobaseTestP(observation=place_holder,
                                name='a Rheobase test')

    dtc.rheobase = None
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    rtest.params['injected_square_current'] = {}
    print('gets here')
    rtest.params['injected_square_current']['delay'] = DELAY
    rtest.params['injected_square_current']['duration'] = DURATION
    dtc.rheobase = rtest.generate_prediction(model)
    if dtc.rheobase is None:
        dtc.rheobase = - 1.0
    return dtc


def dtc_to_rheo_serial(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    dtc.scores = {}
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
            dtc.scores[rtest.name] = 1.0 - score.norm_score
            rtest.params['injected_square_current']['amplitude'] = dtc.rheobase
        else:
            dtc.rheobase = None
            dtc.scores[rtest.name] = 1.0

    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
    return dtc


def substitute_parallel_for_serial(rtest):
    rtest = RheobaseTestP(rtest.observation)
    return rtest

def get_rtest(dtc):
    rtest = None
    place_holder = {}
    place_holder['n'] = 86
    place_holder['mean'] = 10*pq.pA
    place_holder['std'] = 10*pq.pA
    place_holder['value'] = 10*pq.pA

    if not hasattr(dtc,'tests'):#, type(None)):
        if 'RAW' in dtc.backend or 'HH' in dtc.backend or 'GLIF' in dtc.backend:#Backend:
            rtest = RheobaseTest(observation=place_holder,
                                    name='a Rheobase test')
        else:
            rtest = RheobaseTestP(observation=place_holder,
                                    name='a Rheobase test')
    else:
        if 'RAW' in dtc.backend or 'HH' in dtc.backend or 'GLIF' in dtc.backend:
            if not isinstance(dtc.tests, Iterable):
                rtest = dtc.tests
            else:
                #import pdb; pdb.set_trace()
                rtest = [ t for t in dtc.tests if str('RheobaseTest') == t.name ]
                if len(rtest):
                    rtest = rtest[0]
                    #rtest = substitute_parallel_for_serial(rtest[0])
                else:
                    rtest = RheobaseTest(observation=place_holder,
                                            name='a Rheobase test')

        else:
            if not isinstance(dtc.tests, Iterable):
                rtest = dtc.tests
            else:
                rtest = [ t for t in dtc.tests if str('RheobaseTestP') == t.name ]
                if len(rtest):
                    rtest = substitute_parallel_for_serial(rtest[0])
                else:
                    rtest = RheobaseTestP(observation=place_holder,
                                            name='a Rheobase test')
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
    model.attrs = dtc.attrs
    rtest = get_rtest(dtc)
    if rtest is not None:
        dtc.rheobase = rtest.generate_prediction(model)
        #import pdb
        #pdb.set_trace()

        if type(dtc.rheobase) is not type(None):
            if not hasattr(dtc,'prediction'):
                dtc.prediction = {}
            dtc.prediction[str(rtest.name)] = dtc.rheobase
            dtc.rheobase = dtc.rheobase['value']
            obs = rtest.observation
            score = rtest.compute_score(obs,dtc.rheobase)
            if type(dtc.scores) is not type(None):
                dtc.scores[rtest.name] = 1.0 - float(score.norm_score)

            else:
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

def switch_logic(tests):
    # move this logic into sciunit tests
    '''
    Hopefuly depreciated by future NU debugging.
    '''
    if not isinstance(tests,Iterable):
        if str('RheobaseTest') == tests.name:
            active = True
            passive = False
        elif str('RheobaseTestP') == tests.name:
            active = True
            passive = False
    else:
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

def active_values(keyed,rheobase,square = None):
    keyed['injected_square_current'] = {}
    if square == None:
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
            print(dtc.vtest[k]['injected_square_current']['delay']+dtc.vtest[k]['injected_square_current']['duration'])
        elif v.passive == True and v.active == False:
            keyed = dtc.vtest[k]
            dtc.vtest[k] = passive_values(keyed)
    return dtc



def allocate_worst(tests,dtc):
    # If the model fails tests, and cannot produce model driven data
    # Allocate the worst score available.
    if not hasattr(dtc,'scores'):
        dtc.scores = {}
    if type(dtc.scores) is type(None):
        dtc.scores = {}
    for t in tests:
        dtc.scores[str(t.name)] = 1.0
    print(np.sum(list(dtc.scores.values())),len(dtc.tests))
    #import pdb; pdb.set_trace()
    #assert len(dtc.tests) == np.sum(list(dtc.scores.values()))
    return dtc


def pred_evaluation(dtc):
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
    #preds = []
    dtc.preds = None
    dtc.preds = {}
    dtc =  dtc_to_rheo(dtc)
    dtc = format_test(dtc)
    tests = dtc.tests


    for k,t in enumerate(tests):
        if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
            t.params = dtc.vtest[k]
            test_and_models = (t, dtc)
            pred = pred_only(test_and_models)
            dtc.preds[t] = pred

        else:
            dtc.preds[t] = dtc.rheobase
    return dtc

def nunit_evaluation_simple(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    tests = dtc.rtests
    dtc = copy.copy(dtc)
    if not hasattr(dtc,'scores') or dtc.scores is None:
        dtc.scores = None
        dtc.scores = {}


    for k, t in enumerate(tests):
        key = str(t)
        #if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
        dtc.scores[key] = 1.0
        t.params = dtc.vtest[k]

        score, dtc = bridge_judge((t, dtc))
        if score is not None:
            if score.norm_score is not None:
                assignment = 1.0 - score.norm_score
                dtc.scores[key] = assignment
        else:
            print(score,t.name)

            dtc.scores[key] = 1.0
            dtc = allocate_worst(tests, dtc)

            #print('gets here',t.name)
        #print(dtc.scores[key])

    # compute the sum of sciunit score components.
    dtc.summed = dtc.get_ss()

    return dtc


def nunit_evaluation(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    tests = dtc.tests
    dtc = copy.copy(dtc)
    if not hasattr(dtc,'scores') or dtc.scores is None:
        dtc.scores = None
        dtc.scores = {}

    try:
        dtc.model_path = path_params['model_path']
    except:
        print('only some models need paths')
    if isinstance(dtc.rheobase,type(None)) or type(dtc.rheobase) is type(None):

        dtc = allocate_worst(tests, dtc)
        # this should happen when a model lacks a feature it is tested on
    else:
        for k, t in enumerate(tests):
            key = str(t)
            if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
                dtc.scores[key] = 1.0
                t.params = dtc.vtest[k]

                score, dtc = bridge_judge((t, dtc))
                if score is not None:
                    if score.norm_score is not None:
                        assignment = 1.0 - score.norm_score
                        dtc.scores[key] = assignment
                else:
                    print(score,t.name)

                    dtc.scores[key] = 1.0
                    dtc = allocate_worst(tests, dtc)

                    print('gets here',t.name)
                print(dtc.scores[key])

    # compute the sum of sciunit score components.
    dtc.summed = dtc.get_ss()

    return dtc




def evaluate(dtc,regularization=False):
    #print(dtc.rheobase)
	# compute error using L2 regularization.
    if len(dtc.tests) >= 9:
        error_length = len(dtc.tests)
    else:
        error_length = 9 # len(dtc.scores.keys())
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    fitness = [ 1.0 for i in range(0,error_length) ]
    for k,t in enumerate(dtc.scores.keys()):
       if regularization == True:
         fitness[k] = dtc.scores[str(t)]**(1.0/2.0)
       else:
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



def scale(X):
    before = copy.copy(X)
    for i in range(0,np.shape(X)[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
    return X, before

def data_versus_optimal(ga_out):
    rts,complete_map = pickle.load(open('../tests/russell_tests.p','rb'))

    dtcpop = [ p.dtc for p in ga_out['pf'] ]
    pop = [ p for p in ga_out['pf'] ]
    # first a nice violin plot of the test data.
    to_norm = np.matrix([list(t.data) for t in tests ])

    X,before = scale(to_norm)

    print(t)
    ax = sns.violinplot(x="test type", y="physical unit", hue="smoker",
                 data=X, palette="muted")

    for t in tests:
        print(t)
        plt.clf()
        fig, ax = plt.subplots()
        if t.name not in ga_out['pf'][0].dtc.prediction.keys():
            print('gets here',t.name)
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
            mode0*qt.unitless
            mode0 = mode0.rescale(opt_value)
            print(mode0-opt_value)
            half = (bins[1]-bins[0])/2.0
            td = sorted(t.data)
            td = [t*qt.unitless for t in td]
            td = [t.rescale(opt_value) for t in td]
        except:
            import pdb; pdb.set_trace()
        print(sorted(t.data)[-1],'data')
        print(opt_value,'optimizer')
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
    delta_y = Y_test - sklearn_sgd_predictions;
    #pl.matshow(cm)
    #pl.title('Confusion matrix of the classifier')
    #pl.colorbar()
    #pl.show()

    return (sgd,losslasso,lossridge)

def run_ga(explore_edges, max_ngen, test, free_params = None, hc = None, NSGA = None, MU = None, seed_pop = None, model_type = str('RAW')):
    # seed_pop can be used to
    # to use existing models, that are good guesses at optima, as starting points for optimisation.
    # https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python
    # These imports need to be defined with local scope to avoid circular importing problems
    # Try to fix local imports later.
    from neuronunit.optimisation.optimisations import SciUnitOptimisation

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
    DO = SciUnitOptimisation(offspring_size = MU, error_criterion = test, boundary_dict = ss, backend = model_type, hc = hc, selection = selection)#,, boundary_dict = ss, elite_size = 2, hc=hc)


    if seed_pop is not None:
        # This is a re-run condition.
        DO.setnparams(nparams = len(free_params), boundary_dict = ss)

        DO.seed_pop = seed_pop
        DO.setup_deap()

    ga_out = DO.run(max_ngen = max_ngen)
    #import pdb; pdb.set_trace()


    return ga_out, DO



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

def obtain_rheobase(pop, td, tests):
    '''
    Calculate rheobase for a given population pop
    Ordered parameter dictionary td
    and rheobase test rt
    '''
    pop, dtcpop = init_pop(pop, td, tests)
    if 'RAW' in dtcpop[0].backend  or 'HH' in dtcpop[0].backend:

        dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
        dtcpop = list(dtcbag.map(dtc_to_rheo))
        dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
        dtcpop = list(dtcbag.map(format_test))
    else:
        dtcpop = list(map(dtc_to_rheo,dtcpop))
        dtcpop = list(map(format_test,dtcpop))

    for ind,d in zip(pop,dtcpop):
        if type(d.rheobase) is not type(None):
            ind.rheobase = d.rheobase
            d.rheobase = d.rheobase
        else:
            ind.rheobase = None
            d.rheobase = None
    return pop, dtcpop

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

def serial_route(pop,td,tests):
    '''
    parallel list mapping only works with an iterable collection.
    Serial route is intended for single items.
    '''
    if type(dtc.rheobase) is type(None):
        print('Error Score bad model')
        for t in tests:
            dtc.scores[t.names] = 1.0
            dtc.get_ss()
    else:
        dtc = format_test((dtc,tests))
        dtc = nunit_evaluation((dtc,tests))

    return pop, dtc

def filtered(pop,dtcpop):
    dtcpop = [ dtc for dtc in dtcpop if type(dtc.rheobase) is not type(None) ]
    pop = [ p for p in pop if type(p.rheobase) is not type(None) ]
    if len(pop) != len(dtcpop):
        import pdb; pdb.set_trace()

    assert len(pop) == len(dtcpop)
    return (pop,dtcpop)


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

def average_measurements(flat_iter):
    dtca,dtcb = flat_iter
    a = OrderedDict()
    b = OrderedDict()
    tests = OrderedDict()
    for k,v in dtcb.preds.items():
        b[k.name] = v
    for k,v in dtca.preds.items():
        a[k.name] = v
    for k,v in dtca.preds.items():
        tests[k.name] = k
    #print(a,b)

    for k in b.keys():
        if isinstance(a[k], type(None)) or isinstance(b[k], type(None)):
            dtcb.scores[k] = dtca.scores[k] = 1.0 #- score.norm_score
            # store the averaged values in the second half of genes.
            #1.0 #- score.norm_score
            dtcb.twin = None
            dtca.twin = None
            dtca.twin = dtcb.attrs
            dtcb.twin = dtca.attrs
            break
        else:
            try:
                aa = which_thing(a[k])
                bb = which_thing(b[k])
            except:
                pass

            t = tests[k]
            cc = which_thing(t.observation)
            try:
                key = which_key(a[k])
            except:
                key = 'value'

            grab_units = cc['standard'].units

            pp = {}
            pp[key] = np.mean([aa['standard'],bb['standard']])*grab_units
            pp['n'] = 1
            cc['n'] = 1
            try:
                score = t.compute_score(cc,pp)
                dtcb.scores[k]  = dtca.scores[k] = 1.0 - score.norm_score

                print(score.norm_score)
            except:
                score = None
                dtcb.scores[k]  = dtca.scores[k] = 1.0

            dtcb.twin = None
            dtca.twin = None
            dtca.twin = dtcb.attrs
            dtcb.twin = dtca.attrs
            # store the averaged values in the first half of genes.
            # store the averaged values in the second half of genes.
    return (dtca,dtcb)


def opt_pair(dtcpop):
    '''
    Used for searching and optimising where averged double sets of model_parameters
    are the most fundamental gene-unit (rather than single points in high dim parameter space).
    In other words what is optimised sampled and explored, is the average of two waveform measurements.
    This allows for the ultimate solution, to be expressed as two disparate parameter points, that when averaged
    produce a good model.
    The motivating argument for doing things this way, is because the models, and the experimental data
    results from averaged contributions of measurements from clustered data points making a model with optimal
    error, theoretically unaccessible.
    '''
    NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])

    # from neuronunit.optimisation.optimisations import SciUnitOptimisation
    # get waveform measurements, and store in genes.
    dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
    dtcpop = list(dtcbag.map(pred_evaluation).compute())
    # divide the genes pool in half
    ab_s = list(itertools.combinations(dtcpop, 2))

    # average measurements between first half of gene pool, and second half.

    dtc_mixed = []
    for pair in ab_s:
        dtc = average_measurements(pair)
        dtc_mixed.append(dtc)
    dtcpopa = [dtc[0] for dtc in dtc_mixed]
    dtcpopb = [dtc[1] for dtc in dtc_mixed]

    dtcpop = dtcpopa
    dtcpop.extend(dtcpopb)
    assert len(dtcpop) == 2*len(dtcpopb)
    return dtcpop

def opt_on_pair_of_points(dtcpop):
    '''
    Used for searching and optimising where averged double sets of model_parameters
    are the most fundamental gene-unit (rather than single points in high dim parameter space).
    In other words what is optimised sampled and explored, is the average of two waveform measurements.
    This allows for the ultimate solution, to be expressed as two disparate parameter points, that when averaged
    produce a good model.
    The motivating argument for doing things this way, is because the models, and the experimental data
    results from averaged contributions of measurements from clustered data points making a model with optimal
    error, theoretically unaccessible.
    '''
    NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])

    # from neuronunit.optimisation.optimisations import SciUnitOptimisation
    # get waveform measurements, and store in genes.
    dtcbag = db.from_sequence(dtcpop, npartitions = NPART)

    dtcpop = list(dtcbag.map(pred_evaluation).compute())
    # divide the genes pool in half
    dtcpopa,dtcpopb = split_list(copy.copy(dtcpop))
    # average measurements between first half of gene pool, and second half.
    flat_iter = zip(dtcpopa,dtcpopb)
    #dtc_mixed = list(map(average_measurements,flat_iter))
    dtcbag = db.from_sequence(flat_iter, npartitions = NPART)
    dtc_mixed = list(dtcbag.map(average_measurements))
    dtcpopa = [dtc[0] for dtc in dtc_mixed]
    dtcpopb = [dtc[1] for dtc in dtc_mixed]

    dtcpop = dtcpopa
    dtcpop.extend(dtcpopb)
    assert len(dtcpop) == 2*len(dtcpopb)
    return dtcpop

def dtc2model(pop,dtcpop):
    fitness_attr = pop[0].fitness
    if type(fitness_attr) is type(None):
        import pdb; pdb.set_trace()
    for i,p in enumerate(pop):
        if not hasattr(p,'fitness'):
            p.fitness = fitness_attr

    for i,ind in enumerate(pop):
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

def boot_new_genes(number_genes,dtcpop,td):
    '''
    Boot strap new genes to make up for completely called onesself.
    '''
    from neuronunit.optimisation.optimisations import SciUnitOptimisation
    import random
    from datetime import datetime
    random.seed(datetime.now())
    #random.seed(64)
    DO = SciUnitOptimisation(offspring_size = number_genes,
    error_criterion = [dtcpop[0].tests], boundary_dict = dtcpop[0].boundary_dict,
     backend = dtcpop[0].backend, selection = str('selNSGA'))#,, boundary_dict = ss, elite_size = 2, hc=hc)
    DO.setnparams(nparams = len(dtcpop[0].attrs), boundary_dict = dtcpop[0].boundary_dict)
    DO.setup_deap()
    pop = []
    if number_genes<5:
        pop = DO.set_pop(boot_new_random=5)
    else:
        pop = DO.set_pop(boot_new_random=number_genes)
    pop = dtc2model(pop,dtcpop)
    dtcpop_ = update_dtc_pop(pop,td)
    dtcpop_ = pop2dtc(pop,dtcpop_)
    dtcpop_ = list(map(dtc_to_rheo,dtcpop_))
    for i,ind in enumerate(pop):
        pop[i].rheobase = dtcpop_[i].rheobase
    pop = pop[0:number_genes]
    dtcpop_ = dtcpop_[0:number_genes]

    return (pop,dtcpop_)

def resample_high_sampling_freq(dtcpop):
    for d in dtcpop:
        d.attrs['dt'] = d.attrs['dt'] *(1.0/21.41)
    import pdb; pdb.set_trace()
    (dtcpop,_) = add_dm_properties_to_cells(dtcpop)
    return dtcpop

def parallel_route(pop,dtcpop,tests,td,clustered=False):
    NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])
    for d in dtcpop:
        d.tests = copy.copy(tests)
    dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
    dtcpop = list(dtcbag.map(format_test).compute())

    #dtcbag = db.from_sequence(copy.copy(dtcpop), npartitions = NPART)
    #stuff = list(dtcbag.map(inject_rh_and_dont_plot).compute())

    #print([d.rheobase for d in dtcpop])
    #for dtc in stuff:
    #    (model,times,vm) = dtc
    #    print('spike count',model.get_spike_count())
    #for dtc in dtcpop:
    #    inject_and_plot(dtc)
    #pdb.set_trace()

    #dtcpop = list(map(nunit_evaluation,dtcpop))

    dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
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
        return (pop,dtcpop)
    if delta:
        cnt = 0
        while delta:
            '''
            ind,dtc = new_single_gene(pop,dtcpop,td)
            print('new genes 1, here they are',ind,dtc)
            if type(dtc.rheobase) is not type(None):
                pop.append(ind)
                dtcpop.append(dtc)
            '''
            pop_,dtcpop_ = boot_new_genes(delta,spare,td)
            print('new genes , here they are',pop_,dtcpop_)
            print('delta,cnt',delta,cnt)
            pop_ = [ p for p in pop_ if len(p)>1 ]
            if cnt>=2 or not len(pop_):
                pop.extend(pop[0])
                dtcpop.extend(dtcpop[0])
            else:
                pop.extend(pop_)
                dtcpop.extend(dtcpop_)
            (pop,dtcpop) = filtered(pop,dtcpop)
            for i,p in enumerate(pop):
                if not hasattr(p,'fitness'):
                    p.fitness = fitness_attr

            after = len(pop)
            delta = before-after
            if not delta:
                break
        return pop, dtcpop

        '''
        except:
            if len(dtcpop) == 0:
                ind,dtc = new_genes(pop_,dtcpop_,td)
            else:
                ind,dtc = new_genes(pop,dtcpop,td)

            if type(dtc.rheobase) is not type(None):
                ind.fitness = fitness_attr
                pop.append(ind)
                dtcpop.append(dtc)
                cnt += 1
                print(cnt,str(' '),dtc.rheobase)

        '''
def grid_search(explore_ranges,test_frame,backend=None):
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
        dtcpop = list(dtcbag.map(format_test))
        dtcpop = [ dtc for dtc in dtcpop if type(dtc.rheobase) is not type(None) ]

        dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
        dtcpop = list(dtcbag.map(nunit_evaluation))
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
            if True == nested_val:
                seed = nested_key
                seeds[k] = seed
    with open(str(backend)+'_seeds.p','wb') as f:
        pickle.dump(seeds,f)
    return seeds, df


def test_runner(pop,td,tests,single_spike=True):
    if single_spike:
        pop_, dtcpop = obtain_rheobase(pop, td, tests)
        pop, dtcpop = make_up_lost(copy.copy(pop_), dtcpop, td)
        # there are many models, which have no actual rheobase current injection value.
        # filter, filters out such models,
        # gew genes, add genes to make up for missing values.
        # delta is the number of genes to replace.

    else:
        pop_, dtcpop = init_pop(pop, td, tests)
        #pop, dtcpop = obtain_rheobase(pop, td, tests)


    pop,dtcpop = parallel_route(pop, dtcpop, tests, td, clustered=False)
    for ind,d in zip(pop,dtcpop):
        ind.dtc = d
        if not hasattr(ind,'fitness'):
            ind.fitness = copy.copy(pop_[0].fitness)
            for i,v in enumerate(list(ind.fitness.values)):
                ind.fitness.values[i] = list(ind.dtc.scores.values())[i]

    return pop,dtcpop

def update_deap_pop(pop, tests, td, backend = None,hc = None,boundary_dict = None):
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
        print('len(pop)==0')
        import pdb
        pdb.set_trace()
    #pop = copy.copy(pop)
    if hc is not None:
        pop[0].td = None
        pop[0].td = td

    if hc is not None:
        pop[0].hc = None
        pop[0].hc = hc

    if backend is not None:
        pop[0].backend = None
        pop[0].backend = backend
    if boundary_dict is not None:
        pop[0].boundary_dict = None
        pop[0].boundary_dict = boundary_dict

    pop, dtcpop = test_runner(pop,td,tests)
    for p,d in zip(pop,dtcpop):
        p.dtc = d

    #print(pop)


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
