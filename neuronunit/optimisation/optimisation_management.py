#import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
import matplotlib
matplotlib.use('agg')
CONFIDENT = False
#    Goal is based on this. Don't optimize to a singular point, optimize onto a cluster.
#    Golowasch, J., Goldman, M., Abbott, L.F, and Marder, E. (2002)
#    Failure of averaging in the construction
#    of conductance-based neuron models. J. Neurophysiol., 87: 11291131.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import dask.bag as db
import pandas as pd
#import dask.bag as db
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
from neuronunit.models.reduced import ReducedModel
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
#try:
#    from neuronunit.tests.base import passive_AMPL, passive_DELAY, passive_DURATION
#except:
#    pass
import efel

from allensdk.ephys.extract_cell_features import extract_cell_features
#from neuronunit.models import VeryReducedModel
#from sciunit.models.runnable import RunnableModel
from neuronunit.models import ReducedModel

from sciunit.models.runnable import RunnableModel
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from collections.abc import Iterable

CONFIDENT = False

#DURATION = 2000
#DELAY = 200
#from allensdk.ephys import ephys_extractor
#EphysSweepSetFeatureExtractor = ephys_extractor.EphysSweepSetFeatureExtractor
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

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

    uc = { 'injected_square_current': {'amplitude':rheobase,'duration':DURATION,'delay':DELAY }}

    dtc.run_number += 1
    model.set_attrs(**dtc.attrs)
    model.inject_square_current(uc['injected_square_current'])
    return (model, model.get_membrane_potential().times,model.get_membrane_potential(),uc)

import seaborn as sns
from neuronunit.capabilities.spike_functions import get_spike_waveforms

def inject_and_plot(dtc,second_pop=None,third_pop=None,figname='problem'):
    sns.set_style("darkgrid")

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
            pass
            #print(str(dtc.backend))
        #vm = model.get_membrane_potential().magnitude
        sns.set_style("darkgrid")
        plt.plot(model.get_membrane_potential().times,model.get_membrane_potential().magnitude)#,label='ground truth')
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

            #NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])

            #dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
            #dtcpop = list(dtcbag.map(nuunit_dm_evaluation).compute())
            #latency = np.mean([dtc.AP1DelayMeanTest for dtc in dtcpop])

            for dtc in dtcpop:
                model = mint_generic_model(dtc.backend)
                try:
                    rheobase = dtc.rheobase['value']
                except:
                    rheobase = dtc.rheobase
                uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                if rheobase is None:
                    break
                model.inject_square_current(uc)
                if model.get_spike_count()>1:
                    break
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    pass
                    #print(str(dtc.backend))
                vm = model.get_membrane_potential()#.magnitude
                #print(np.max(vm))
                #sns.set_style("darkgrid")
                if str("RAW") in dtc.backend:
                    label=str('Izhikevich Model')

                if str("ADEXP") in dtc.backend:
                    label=str('Adaptive Exponential Model')
                if str("GLIF") in dtc.backend:
                    label=str('Generalized Leaky Integrate and Fire')
                #label = label+str(latency)

                sns.set_style("darkgrid")
                #from neuronunit.capabilities import get_spike_waveforms
                snipptes = get_spike_waveforms(vm)
                plt.plot(snipptes.times,snipptes,color='red',label=label)#,label='ground truth')
                ax.legend()
            #second_pop = list(map(nuunit_dm_evaluation,second_pop))


            #NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])

            #dtcbag = db.from_sequence(second_pop, npartitions = NPART)
            #second_pop = list(dtcbag.map(nuunit_dm_evaluation).compute())
            #latency = np.mean([dtc.AP1DelayMeanTest for dtc in second_pop])


            #latency = np.mean([dtc.AP1DelayMeanTest for dtc in second_pop])

            for dtc in second_pop:
                model = mint_generic_model(dtc.backend)
                try:
                    rheobase = dtc.rheobase['value']
                except:
                    rheobase = dtc.rheobase
                if rheobase is None:
                    break
                uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                if model.get_spike_count()>1:
                    break
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    pass
                    #print(str(dtc.backend))
                vm = model.get_membrane_potential()#.magnitude
                if str("RAW") in dtc.backend:
                    label=str('Izhikevich Model')

                if str("ADEXP") in dtc.backend:
                    label=str('Adaptive Exponential Model')

                if str("GLIF") in dtc.backend:
                    label=str('Generalized Leaky Integrate and Fire')
                #label = label+str(latency)

                sns.set_style("darkgrid")
                snipptes = get_spike_waveforms(vm)
                plt.plot(snipptes.times,snipptes,color='blue',label=label)#,label='ground truth')

                #plt.plot(model.get_membrane_potential().times,vm,color='blue',label=label)#,label='ground truth')
                #ax.legend(['A simple line'])
                ax.legend()
            if third_pop is None:
                return
            else:

                #dtcbag = db.from_sequence(third_pop, npartitions = NPART)
                #third_pop = list(dtcbag.map(nuunit_dm_evaluation).compute())
                #latency = np.mean([dtc.AP1DelayMeanTest for dtc in third_pop])

                #third_pop = list(map(nuunit_dm_evaluation,third_pop))
                #latency = np.mean([dtc.AP1DelayMeanTest for dtc in third_pop])

                for dtc in third_pop:
                    model = mint_generic_model(dtc.backend)
                    try:
                        rheobase = dtc.rheobase['value']
                    except:
                        rheobase = dtc.rheobase
                    uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                    if rheobase is None:
                        break
                    dtc.run_number += 1
                    model.set_attrs(**dtc.attrs)
                    model.inject_square_current(uc)
                    if model.get_spike_count()>1:
                        break
                    if str(dtc.backend) in str('ADEXP'):
                        model.finalize()
                    else:
                        pass
                        #print(str(dtc.backend))
                    vm = model.get_membrane_potential()#.magnitude
                    sns.set_style("darkgrid")

                    if str("RAW") in dtc.backend:
                        label=str('Izhikevich Model')

                    if str("ADEXP") in dtc.backend:
                        label=str('Adaptive Exponential Model')
                    if str("GLIF") in dtc.backend:
                        label=str('Generalized Leaky Integrate and Fire')
                    #label = label+str(latency)
                    snipptes = get_spike_waveforms(vm)
                    plt.plot(snipptes.times,snipptes,color='green',label=label)#,label='ground truth')
                    #ax.legend(['A simple line'])
                    ax.legend()
                plot_backend = mpl.get_backend()
                #if plot_backend == str('Agg'):
                plt.title(figname)
                plt.savefig(figname+str('all_traces.png'))
                #else:
                #    plt.show()
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
    #if model.get_membrane_potential() is not None:
    #    return (model.get_membrane_potential().times,model.get_membrane_potential())
    #else:
    return (None,None)
def make_imputed_observations(tests,backend,random_param):
    '''
    to be used in conjunction with round_trip_test below.

    '''
    dtc = DataTC()
    dtc.attrs = random_param
    dtc.backend = backend
    #dtc.tests  = tests
    #import pdb; pdb.set_trace()
    if str('RheobaseTest') in tests.keys():
        dtc = get_rh(dtc,tests['RheobaseTest'])
    tests = list(tests.values())
    dtc.tests = tests
    dtc = pred_evaluation(dtc)
    print('gets here d',dtc)

    # Temporarily patch sciunit judge code, which seems to be broken.
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
        print('gets here b')
        ga_out, DO = run_ga(explore_param,NGEN,tests,free_params=free_params, NSGA = True, MU = MU, backed=backend, selection=str('selNSGA2'),seed_pop=pf[0].dtc.attrs)
        best = ga_out['pf'][0].dtc.get_ss()
    #print('Its ',Bool(best < 0.5), ' that optimisation succeeds on this model class')
    #print('goodness of fit: ',best)
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

def mint_generic_model(backend):
    LEMS_MODEL_PATH = path_params['model_path']
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = str(backend))
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
    model.set_attrs(**dtc.attrs)
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
        #if str('Rheobase') in str(test.name) and str(dtc.backend) in str('GLIF') :
        #    pass
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

    if 'RAW' in backend_ or 'HH' in backend_:#_ or 'ADEXP' in backend_ or 'GLIF' in backend_:#Backend:
        rtest = RheobaseTest(observation=place_holder,
                                name='a Rheobase test')
    else:
        rtest = RheobaseTestP(observation=place_holder,
                                name='a Rheobase test')

    dtc.rheobase = None
    model = mint_generic_model(backend_)
    #import pdb; pdb.set_trace()
    model.set_attrs(**dtc.attrs)
    rtest.params['injected_square_current'] = {}
    #print('gets here')
    rtest.params['injected_square_current']['delay'] = DELAY
    rtest.params['injected_square_current']['duration'] = DURATION
    dtc.rheobase = rtest.generate_prediction(model)
    if dtc.rheobase is None:
        dtc.rheobase = - 1.0
    return dtc


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
        if 'RAW' in dtc.backend or 'HH' in dtc.backend:# or 'GLIF' in dtc.backend:#Backend:
            rtest = RheobaseTest(observation=place_holder,
                                    name='a Rheobase test')
        else:
            rtest = RheobaseTestP(observation=place_holder,
                                    name='a Rheobase test')
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
                                                name='a Rheobase test')
                    #rtest = substitute_parallel_for_serial(rtest[0])
                else:
                    rtest = RheobaseTest(observation=place_holder,
                                            name='a Rheobase test')
            #except:
                #    rtest = RheobaseTest(observation=place_holder,
                #                            name='a Rheobase test')

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
                                                name='a Rheobase test')
                else:
                    rtest = RheobaseTestP(observation=place_holder,
                                            name='a Rheobase test')

                #rtest = [ t for t in dtc.tests if str('RheobaseTestP') == t.name ]
                #if len(rtest):
                #    rtest = substitute_parallel_for_serial(rtest[0])
                #else:

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
        if isinstance(rtest,Iterable):
            #import pdb; pdb.set_trace()
            rtest = rtest[0]
        dtc.rheobase = rtest.generate_prediction(model)
        if type(dtc.rheobase) is not type(None):
            if not hasattr(dtc,'prediction'):
                dtc.prediction = {}
            dtc.prediction[str(rtest.name)] = dtc.rheobase
            dtc.rheobase = dtc.rheobase['value']
            obs = rtest.observation
            score = rtest.compute_score(obs,dtc.rheobase)
            if type(score.norm_score) is not type(None):
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
        #import pdb; pdb.set_trace()
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
            #print(dtc.vtest[k]['injected_square_current']['delay']+dtc.vtest[k]['injected_square_current']['duration'])
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

    #assert len(dtc.scores) == dtc.error_length
    #assert np.sum(dtc.scores.values) == dtc.error_length
    #print(np.sum(list(dtc.scores.values())),len(dtc.tests))
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
    dtc = dtc_to_rheo(dtc)

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

            dtc.scores[key] = 1.0
            dtc = allocate_worst(tests, dtc)
    # compute the sum of sciunit score components.
    dtc.summed = dtc.get_ss()

    return dtc

from neuronunit.tests import dm_test_interoperable #import Interoperabe
from neuronunit.tests.base import VmTest
from scipy.signal import decimate
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
global cpucount
cpucount = multiprocessing.cpu_count()
from scipy.special import logit as logistic

#from neuronunit.examples.hide_imports import *
#print(os.cwd())
#import pdb
#pdb.set_trace()
rts,complete_map = pickle.load(open('../../tests/russell_tests.p','rb'))
df = pd.DataFrame(rts)
for key,v in rts.items():
    helper_tests = [value for value in v.values() ]
    break

def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))

def current_and_features(dtc):
    #dtc.rheobase*3.0
    #current = {'injected_square_current':
    #            {'amplitude':pre_obs[1]*pq.pA, 'delay':DELAY, 'duration':DURATION}}
    current = {'injected_square_current':
                {'amplitude':dtc.ampl, 'delay':DELAY, 'duration':DURATION}}
    compare = dtc.pre_obs
    target = len(dtc.pre_obs['spikes'])

    model = mint_generic_model(dtc.backend)
    model.set_attrs(**dtc.attrs)
    #model.name = str(pre_obs[0])

    vm30 = model.inject_square_current(current['injected_square_current'])
    vm30 = model.get_membrane_potential()

    if model.get_spike_count()!=target:
        '''
        rejection due to not meeting target
        '''
        dtc.scores = None
        dtc.fitted_spike_cnt = False
        #return dtc
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

        #t = decimate(t,13)
        #v = decimate(v,13)
        #vm30 = model.inject_square_current(current['injected_square_current'])
        if dtc.backend in str('ADEXP'):
            vm30 = model.finalize()
            #v = [float(v)*1000.0
            v = [ float(v*1000.0) for v in vm30.magnitude]
            t = [ float(t) for t in vm30.times ]
        #t = decimate(t,13)
        #v = decimate(v,13)
        try:
            ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
            ephys.process_spikes()
        except:
            return dtc
    ephys_dict = ephys.as_dict()

    dtc.scores = None
    dtc.scores = {}
    #test = copy.copy(helper_tests[0])
    #compare = pre_obs
    #import pdb; pdb.set_trace()
    for k,v in compare.items():
        dtc.scores[k] = 1.0
    helper = helper_tests[0]

    for k,observation in compare.items():
        if str(k) not in str('spikes'):
            '''
            compute interspike firing frame_dynamics
            '''

            obs = {}
            obs['mean'] = observation
            prediction = {}
            if k in ephys_dict.keys():
                prediction['mean'] = ephys_dict[k]
                helper.name = str(k)
                obs['std']=1.0
                prediction['std']=1.0
                score = VmTest.compute_score(helper,obs,prediction)
                if score is not None and score.norm_score is not None:
                    dtc.scores[k] = 1.0-score.norm_score
                else:
                    dtc.scores[k] = 1.0
        if str(k) in str('spikes'):
            '''
            compute perspike waveform features on just the first spike
            '''
            first_spike = observation[0]
            for key,spike_obs in first_spike.items():

                #if not str('direct') in key and not str('adp_i') in key and not str('peak_i') in key and not str('fast_trough_i') and not str('fast_trough_i') and not str('trough_i'):
                try:
                    obs = {}
                    obs['mean'] = spike_obs
                    prediction = {}
                    prediction['mean'] = ephys_dict['spikes'][0][key]
                    helper.name = str(key)
                    obs['std']=1.0
                    prediction['std']=1.0
                    score = VmTest.compute_score(helper,obs,prediction)
                    if not score is None and not score.norm_score is None:
                        dtc.scores[key] = 1.0-score.norm_score
                except:
                    dtc.scores[key] = 1.0
    print(dtc.scores)
    dtc.ephys = None
    dtc.ephys = ephys
    spike_cnt_pred = model.get_spike_count()
    delta = np.sqrt(spike_cnt_pred)-np.sqrt(target)


    #import numpy as np

    dtc.scores['spk_count'] = sigmoid(np.abs(delta))
    print(dtc.scores['spk_count'],'spike count delta')
    delta = float(dtc.ampl)-dtc.pre_obs['current_test']
    dtc.scores['current_test'] = sigmoid(np.abs(delta))
    print(dtc.scores['current_test'],'current  injection delta')

    return dtc

def nuunit_allen_evaluation(dtc):

    if hasattr(dtc,'vtest'):
        values = [v for v in dtc.vtest.values()]
        for value in values:
            if str('injected_square_current') in value.keys():
                current = value['injected_square_current']
                current['amplitude'] = dtc.rheobase * 3.0
                break
    else:
        #compare = dtc.pre_obs
        target_spikes = len(dtc.pre_obs['spikes'])
        observation_spike={}
        observation_spike['value']=target_spikes
        from neuronunit.tests.fi import SpikeCountSearch
        scs = SpikeCountSearch(observation_spike)
        target_current = scs.generate_prediction(model)
        steps = np.linspace(float(dtc.rheobase),1.25*float(dtc.rheobase),cpucount+1)*pq.pA
        steps = steps[1:-1]*pq.pA
        current_steps = steps
        dtc_clones = [ copy.copy(dtc) for i in range(0,len(current_steps)) ]
        #import pdb; pdb.set_trace()

        population = []
        for i,s in enumerate(current_steps): dtc_clones[i].ampl = copy.copy(current_steps[i])
        for i,s in enumerate(current_steps):
             dtc = current_and_features(dtc_clones[i])
             population.append(dtc)
        dtcpop = sorted([(p.scores['spk_count'],p) for p in population ], key=lambda tup: tup[0])
             #if dtc.fitted_spike_cnt == True:
        #import pdb; pdb.set_trace()

        return dtcpop[0][1]




from neuronunit.capabilities import spike_functions
import allensdk.ephys.ephys_features as ft
from scipy.signal import decimate
import asciiplotlib as apl
import allensdk.ephys.ephys_features as ft
#import asciiplotlib as apl

def nuunit_dm_evaluation(dtc):
    model = mint_generic_model(dtc.backend)
    model.set_attrs(**dtc.attrs)
    values = [v for v in dtc.vtest.values()][0]
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
        return dtc
    model.vm30 = None
    model.vm30 = vm30
    model.druckmann2013_strong_current = None
    model.druckmann2013_strong_current = dtc.rheobase * 3.0

    model.druckmann2013_input_resistance_currents =[ -5.0*pq.pA, -10.0*pq.pA, -15.0*pq.pA]#,copy.copy(current)

    DMTNMLO = dm_test_interoperable.DMTNMLO()
    DMTNMLO.test_setup(None,None,model= model)
    dm_test_features = DMTNMLO.runTest()
    dtc.AP1DelayMeanTest = None
    dtc.AP1DelayMeanTest = dm_test_features['AP1DelayMeanTest']
    print(dtc.AP1DelayMeanTest)
    dtc.dm_test_features = None
    dtc.dm_test_features = dm_test_features
    return dtc
    '''
    try:
        fig = apl.figure()
        fig.plot(t, v, label=str('spikes: '), width=100, height=20)
        fig.show()
    except:
        pass
    v = [float(v)*1000.0 for v in vm30.magnitude]
    t = [float(t) for t in vm30.times]
    from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

    ephys = EphysSweepFeatureExtractor(t=np.array(t),v=np.array(v))#,\
    ephys.process_spikes()
    print(ephys.spikes(),'spikes')
    spks = ft.detect_putative_spikes(np.array(v),np.array(t))
    print(spks)
    dtc.spks = None
    dtc.spks = spks
    dtc.ephys = None
    dtc.ephys = ephys
    return dtc
    '''

        #,end=finish)

    #current_inh['duration'] = 500.0*pq.ms
    #current_inh['delay'] = 200.0*pq.ms
    #current_inh['amplidtude'] = -10.0*pq.pA
    #self.params['injected_square_current'] = {'amplitude': -10.0*pq.pA,

    #print(dtc.dm_test_features.keys())
    #print(dm_test_features)
    #import pdb; pdb.set_trace()
    ##
    # Wrangle data to prepare for EFEL feature calculation.
    ##
    '''
    values = [v for v in dtc.vtest.values()][0]
    current = values['injected_square_current']

    current['amplidtude'] = dtc.rheobase * 1.5
    model.protocol = None
    model.protocol = {}
    model.protocol['Time_End'] = current['duration'] + current['delay']
    model.protocol['Time_Start'] = current['delay']

    trace3 = {}
    trace3['T'] = [ float(t) for t in model.vm30.times.rescale('ms') ]
    trace3['V'] = [ float(v) for v in model.vm30.magnitude]#temp_vm
    trace3['stimulus_current'] = [ model.druckmann2013_strong_current ]
    trace3['stim_end'] = [ trace3['T'][-1] ]

    #trace3['stim_end'] = [ float(model.protocol['Time_End'])*1000.0 ]
    trace3['stim_start'] = [ trace3['T'][0] ]

    traces3 = [trace3]# Now we pass 'traces' to the efel and ask it to calculate the feature# values

    trace15 = {}
    trace15['T'] = [ float(t) for t in model.vm15.times.rescale('ms') ]
    trace15['V'] = [ float(v) for v in model.vm15.magnitude ]#temp_vm

    #trace15['stim_end'] = [ float(model.protocol['Time_End'])*1000.0 ]
    #trace15['stim_start'] = [ float(model.protocol['Time_Start'])*1000.0 ]
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
    print(efel_30)
    print(efel_15)
    dtc.efel_15 = None
    dtc.efel_30 = None
    dtc.efel_15 = efel_15
    dtc.efel_30 = efel_30
    '''



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
        pass
        #print('only some models need paths')
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
                    dtc.scores[key] = 1.0
                    dtc = allocate_worst(tests, dtc)

    # compute the sum of sciunit score components.
    #delta = dtc.error_length - len(list(dtc.scores.values()))
    #print(delta)


    #for i in range(0,delta):
    #    dtc.scores[i]=1.0 #make up for missing fitness values
    #print('failed to make score length the same')
    print(dtc.scores)
    dtc.summed = dtc.get_ss()
    try:
        greatest = np.max([dtc.error_length,len(dtc.scores)])
    except:
        greatest = len(dtc.scores)
    dtc.scores_ratio = None
    dtc.scores_ratio = dtc.summed/greatest
    return dtc


def evaluate_allen(dtc,regularization=True):
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    #print({k:v for k,v in dtc.scores.items()})
    #greatest = np.max([dtc.error_length,len(dtc.ascores)])
    print(dtc.ascores)
    fitness = [ 1.0 for i in range(0,len(dtc.ascores)) ]
    for int_,t in enumerate(dtc.ascores.keys()):
       if regularization == True:
          if dtc.ascores[str(t)] is None:
              fitness[int_] = 1.0
          else:
              fitness[int_] = dtc.ascores[str(t)]**(1.0/2.0)
       else:
          if dtc.ascores[str(t)] is None:
              fitness[int_] = 1.0
          else:
              fitness[int_] = dtc.ascores[str(t)]
    print(fitness)
    return tuple(fitness,)

def evaluate(dtc,regularization=True):
    # assign worst case errors, and then over write them with situation informed errors as they become available.
    #print({k:v for k,v in dtc.scores.items()})
    if not hasattr(dtc,str('scores')):
        fitness = [ 1.0 for i in range(0,dtc.error_length) ]
        return fitness

    greatest = np.max([dtc.error_length,len(dtc.scores)])
    #import pdb; pdb.set_trace()

    fitness = [ 1.0 for i in range(0,greatest) ]
    for int_,t in enumerate(dtc.scores.keys()):
       if regularization == True:
          fitness[int_] = dtc.scores[str(t)]**(1.0/2.0)
       else:
          fitness[int_] = dtc.scores[str(t)]
    print(fitness)
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
    if 'RAW' in dtcpop[0].backend  or 'HH' in dtcpop[0].backend or str('ADEXP') in dtcpop[0].backend:
        dtcpop = list(map(dtc_to_rheo,dtcpop))
        #dtcpop = list(map(format_test,dtcpop))
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
                dtcb.nuunit_allen_evaluation[k]  = dtca.evaluate[k] = 1.0

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
            # print('last change.')
            #pop[i] = WSListIndividual(pop[i])
            pop[i].dtc = None
        d.get_ss()
        pop[i].dtc = copy.copy(d)
    return dtcpop,pop
def get_dm(pop,dtcpop,tests,td):
    if CONFIDENT == True:
        dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
        dtcpop = list(dtcbag.map(nuunit_dm_evaluation).compute())
    else:
        dtcpop = list(map(nuunit_dm_evaluation,dtcpop))
    dtcpop,pop = score_attr(dtcpop,pop)
    return dtcpop,pop
def get_allen(pop,dtcpop,tests,td):
    NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])
    for dtc in dtcpop:
        dtc.pre_obs = None
        dtc.pre_obs = tests

    if CONFIDENT == True:
        dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
        dtcpop = list(dtcbag.map(nuunit_allen_evaluation).compute())

    else:
        dtcpop = list(map(nuunit_allen_evaluation,dtcpop))
    #dtcpop,pop = score_attr(dtcpop,pop)
    pop = [pop[i] for i,d in enumerate(dtcpop) if type(d) is not type(None)]
    dtcpop = [d for d in dtcpop if type(d) is not type(None)]

    return pop, dtcpop

def parallel_route(pop,dtcpop,tests,td,protocol=None):
    NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])
    #import pdb; pdb.set_trace()
    if protocol['allen'] == True:
        pop, dtcpop = get_allen(pop,dtcpop,tests,td)

        pop = [pop[i] for i,d in enumerate(dtcpop) if type(d) is not type(None)]
        dtcpop = [d for d in dtcpop if type(d) is not type(None)]
        return pop, dtcpop

    elif protocol['dm'] == True:
        pop, dtcpop = get_dm(pop,dtcpop,tests,td)
        return pop, dtcpop

    elif protocol['elephant'] == True:
        #dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
        #dtcpop = list(dtcbag.map(format_test))
        for d in dtcpop:
            d.tests = copy.copy(tests)
        dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
        dtcpop = list(dtcbag.map(format_test).compute())

        if CONFIDENT == True:
            dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
            dtcpop = list(dtcbag.map(nunit_evaluation).compute())
        else:
            dtcpop = list(map(nunit_evaluation,dtcpop))

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
            #ind,dtc = new_single_gene(pop,dtcpop,td)
            pop_,dtcpop_ = boot_new_genes(delta,spare,td)
            for dtc,ind in zip(pop_,dtcpop_):
                ind.from_imputation = None
                dtc.from_imputation = None
                ind.from_imputation = True
                dtc.from_imputation = True
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
            print(cnt,delta,'make new genes')
        return pop, dtcpop

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


def test_runner(pop,td,tests,single_spike=True,protocol=None):
    if single_spike:
        pop_, dtcpop = obtain_rheobase(pop, td, tests)

        for ind,dtc in zip(pop,dtcpop):
            dtc.error_length = ind.error_length
            error_length = ind.error_length
        pop, dtcpop = make_up_lost(copy.copy(pop_), dtcpop, td)

        # there are many models, which have no actual rheobase current injection value.
        # filter, filters out such models,
        # gew genes, add genes to make up for missing values.
        # delta is the number of genes to replace.

    else:

        pop, dtcpop = init_pop(pop, td, tests)
        for ind,dtc in zip(pop,dtcpop):
            dtc.error_length = ind.error_length
            error_length = ind.error_length

        #pop, dtcpop = obtain_rheobase(pop, td, tests)

        for ind,d in zip(pop,dtcpop):
            d.error_length = error_length
            ind.error_length = error_length

    pop,dtcpop = parallel_route(pop, dtcpop, tests, td,protocol=protocol)#, clustered=False)
    for ind,d in zip(pop,dtcpop):
        ind.dtc = None
        ind.dtc = d
        #import pdb; pdb.set_trace()
        print(d.get_ss())
        if not hasattr(ind,'fitness'):
            ind.fitness = copy.copy(pop_[0].fitness)
            for i,v in enumerate(list(ind.fitness.values)):
                ind.fitness.values[i] = list(ind.dtc.evaluate.values())[i]
    pop = [ ind for ind,d in zip(pop,dtcpop) if d.scores is not None ]
    dtcpop = [ d for ind,d in zip(pop,dtcpop) if d.scores is not None ]
    #import pdb; pdb.set_trace()
    return pop,dtcpop


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
        print('len(pop)==0')
        import pdb
        pdb.set_trace()
    #pop = copy.copy(pop)
    if hc is not None:
        pop[0].td = None
        pop[0].td = self.td

    if hc is not None:
        pop[0].hc = None
        pop[0].hc = hc

    if backend is not None:
        pop[0].backend = None
        pop[0].backend = backend
    if boundary_dict is not None:
        pop[0].boundary_dict = None
        pop[0].boundary_dict = boundary_dict
    for p in pop:
        if error_length is not None:
            p.error_length = None
            p.error_length = error_length


    pop, dtcpop = test_runner(pop,td,tests)
    for p,d in zip(pop,dtcpop):
        p.dtc = d
        print(p.error_length)
    return pop

class OptMan():
    def __init__(self,tests, td=None, backend = None,hc = None,boundary_dict = None, error_length=None,protocol=None):
        self.tests = tests
        self.td = td
        self.protocol=protocol
        self.error_length = error_length
        self.tests = tests
        self.backend = backend
        self.hc = hc
        self.boundary_dict= boundary_dict
        self.protocol = protocol
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
        from neuronunit.optimisation.exhaustive_search import update_dtc_grid
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
        if 'RAW' in self.backend  or 'HH' in self.backend or str('ADEXP') in self.backend:
            dtcpop = list(map(dtc_to_rheo,dtcpop))
            #dtcpop = list(map(format_test,dtcpop))
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

    def test_runner(self,pop,td,tests,single_spike=True,protocol=None):
        if single_spike:
            pop_, dtcpop = self.obtain_rheobase(pop, td, tests)

            for ind,dtc in zip(pop,dtcpop):
                dtc.error_length = self.error_length
                error_length = self.error_length
            pop, dtcpop = make_up_lost(copy.copy(pop_), dtcpop, td)

            # there are many models, which have no actual rheobase current injection value.
            # filter, filters out such models,
            # gew genes, add genes to make up for missing values.
            # delta is the number of genes to replace.

        else:

            pop, dtcpop = init_pop(pop, td, tests)
            for ind,dtc in zip(pop,dtcpop):
                dtc.error_length = self.error_length
                error_length = self.error_length

            for ind,d in zip(pop,dtcpop):
                d.error_length = self.error_length
                ind.error_length = self. error_length

        pop,dtcpop = parallel_route(pop, dtcpop, tests, td,protocol=protocol)#, clustered=False)
        for ind,d in zip(pop,dtcpop):
            ind.dtc = None
            ind.dtc = d
            #import pdb; pdb.set_trace()
            print(d.get_ss())
            if not hasattr(ind,'fitness'):
                ind.fitness = copy.copy(pop_[0].fitness)
                for i,v in enumerate(list(ind.fitness.values)):
                    ind.fitness.values[i] = list(ind.dtc.evaluate.values())[i]
        pop = [ ind for ind,d in zip(pop,dtcpop) if d.scores is not None ]
        dtcpop = [ d for ind,d in zip(pop,dtcpop) if d.scores is not None ]
        #import pdb; pdb.set_trace()
        return pop,dtcpop

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
        for p in pop:
            if error_length is not None:
                self.error_length = None
                self.error_length = error_length


        pop, dtcpop = self.test_runner(pop,td,tests,protocol=self.protocol)
        for p,d in zip(pop,dtcpop):
            p.dtc = d
            p.error_length = self.error_length
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
