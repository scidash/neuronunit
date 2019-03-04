#import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
#matplotlib.use('agg')

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
from neuronunit.tests.druckmann2013 import *

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
    try:
        dtc.vtest[0]['injected_square_current']['amplitude'] = dtc.rheobase['value']
    except:
        dtc.vtest[0]['injected_square_current']['amplitude'] = dtc.rheobase
    dtc.vtest[0]['injected_square_current']['duration'] = 1000*pq.ms
    model.inject_square_current(dtc.vtest[0])

    plt.plot(model.get_membrane_potential().times,model.get_membrane_potential())#,label='ground truth')
    plot_backend = mpl.get_backend()

    if plot_backend == str('Agg'):
        plt.savefig(figname+str('debug.png'))
    else:
        plt.show()
'''
def obtain_predictions(t,backend,params):

    model = None
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
    model.set_attrs(params)
    return t.generate_prediction(model)
'''
def init_dm_tests(standard,strong):
    '''
    Lets not greedy import
    import neuronunit.tests.druckmann2013 as dm
    for t in dir(dm):
        if "Test" in t:
            exec('from neuronunit.tests.druckmann2013 import '+str(t))#+' as '+str(t))
    '''

    tests = [AP12AmplitudeDropTest(standard),
        AP1SSAmplitudeChangeTest(standard),
        AP1AmplitudeTest(standard),
        AP1WidthHalfHeightTest(standard),
        AP1WidthPeakToTroughTest(standard),
        AP1RateOfChangePeakToTroughTest(standard),
        AP1AHPDepthTest(standard),
        AP2AmplitudeTest(standard),
        AP2WidthHalfHeightTest(standard),
        AP2WidthPeakToTroughTest(standard),
        AP2RateOfChangePeakToTroughTest(standard),
        AP2AHPDepthTest(standard),
        AP12AmplitudeChangePercentTest(standard),
        AP12HalfWidthChangePercentTest(standard),
        AP12RateOfChangePeakToTroughPercentChangeTest(standard),
        AP12AHPDepthPercentChangeTest(standard),
        AP1DelayMeanTest(standard),
        AP1DelaySDTest(standard),
        AP2DelayMeanTest(standard),
        AP2DelaySDTest(standard),
        Burst1ISIMeanTest(standard),
        Burst1ISISDTest(standard),
        InitialAccommodationMeanTest(standard),
        SSAccommodationMeanTest(standard),
        AccommodationRateToSSTest(standard),
        AccommodationAtSSMeanTest(standard),
        AccommodationRateMeanAtSSTest(standard),
        ISICVTest(standard),
        ISIMedianTest(standard),
        ISIBurstMeanChangeTest(standard),
        SpikeRateStrongStimTest(strong),
        AP1DelayMeanStrongStimTest(strong),
        AP1DelaySDStrongStimTest(strong),
        AP2DelayMeanStrongStimTest(strong),
        AP2DelaySDStrongStimTest(strong),
        Burst1ISISDStrongStimTest(strong),
        Burst1ISIMeanStrongStimTest(strong)]

    AHP_list = [AP1AHPDepthTest(standard),
        AP2AHPDepthTest(standard),
        AP12AHPDepthPercentChangeTest(standard) ]
    return tests

def cell_to_test_mapper(content):
    dm_properties = {}
    index,dtc = content
    dm_properties[index] = []
    ir_currents = dtc.rheobase
    standard = 1.5*ir_currents
    #standard *= 1.5
    strong = 3*ir_currents
    tests = init_dm_tests(standard,strong)
    model = None
    model = mint_generic_model(dtc.backend)

    for i, test in enumerate(tests):
        print(i,test)
        model.set_attrs(dtc.attrs)
        dm_properties[index].append(test.generate_prediction(model)['mean'])
        print(dm_properties[index])
    dtc.dm_properties = None
    dtc.dm_properties = dm_properties
    return dtc

def add_dm_properties_to_cells(dtcpop):
    dm_properties = {}
    flatten_cells = [(index,dtc) for index,dtc in enumerate(dtcpop) ]
    bag = db.from_sequence(flatten_cells,npartitions=8)
    dtcpop = list(bag.map(cell_to_test_mapper).compute())
    dm_properties = {}
    for l in list_of_dics:
        dm_properties.update(l)
    return (dtcpop,dm_properties)


def make_fake_observations(tests,backend,random_param):
    '''
    to be used in conjunction with round_trip_test below.

    '''
    dtc = DataTC()
    dtc.attrs = random_param
    dtc.backend = backend
    dtc = get_rh(dtc,tests[0])

    #pt = format_params(tests,rheobase)
    dtc = pred_evaluation(dtc)

    #ptbag = db.from_sequence(pt[1::])
    #test_and_models = (test, dtc)
    #def pred_only(test_and_models):
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

from neuronunit.optimisation.model_parameters import MODEL_PARAMS
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
        print(ranges[k])
        print(type(ranges[k]))
        #import pdb; pdb.set_trace()
        try:
            mean = np.mean(ranges[k])
            std = np.std(ranges[k])
            sample = numpy.random.normal(loc=mean, scale=2*std, size=1)[0]
            random_param[k] = sample
        except:
            print('probably a list and not a flexible parameter')
            random_param[k] = ranges[k]
    tests = make_fake_observations(tests,backend,random_param)
    NGEN = 10
    MU = 6
    ga_out, DO = run_ga(explore_param,NGEN,tests,free_params=free_params, NSGA = True, MU = MU, backed=backend, selection=str('selNSGA2'))
    best = ga_out['pf'][0].dtc.get_ss()
    print(Bool(best < 0.5))
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


def cluster_tests(use_test,backend,explore_param):
    '''
    Given a group of conflicting NU tests, quickly exploit optimisation, and variance information
    To find subsets of tests that don't conflict.
    Inputs:
        backend string, signifying model backend type
        explore_param list of dictionaries, of model parameter limits/ranges.
        use_test, a list of tests per experimental entity.
    Outputs:
        lists of groups of less conflicted test classes.
        lists of groups of less conflicted test class names.
    '''
    td, test_opt, centres = get_centres(use_test,backend,explore_param)
    cell_attrs = [ centers[:, 0],centers[:, 1], centers[:, 2] ]
    for key, use_test in test_frame.items():
        preds = []
        mean_scores = []
        '''

        Create a model situation analogous to the NeuroElectro data situation.
        Assume, that I have three or more clustered experimental observations,
        averaging wave measurements is inappropriate, but thats what I have.
        Try to reconstruct the clustered means, by clustering solution sets with respect to 8 waveform measurements.
        The key is to realize, that averaging measurements, and computing error is very different to, taking measurements, and averaging error.
        The later is the multiobjective approach to optimisation. The former is the approach used here.
        '''
        dtcpop = []
        for ca in cell_attrs:
            '''
            What is the mean waveform measurement resulting from averaging over the three data points?
            '''
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
        #model_attrs = [ dtc.attrs for dtc in dtcpop ]
        mean_score,dtc = score_only(dtc,mean_pred,test)
        mean_scores.append(mean_score)
        print(mean_scores[-1])

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
    #obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    pred = test.generate_prediction(model)
    return pred#(pred,obs)

def which_thing(thing):
    #print(thing)
    #print(thing.keys())
    if 'value' in thing.keys():
        standard = thing['value']
    if 'mean' in thing.keys():
        standard = thing['mean']
    thing['standard'] = standard
    return thing

def which_key(thing):
    #print(thing)
    #print(thing.keys())
    if 'value' in thing.keys():
        return 'value'
    if 'mean' in thing.keys():
        return 'mean'

def bridge_judge(test_and_dtc):
    # Temporarily patch sciunit judge code, which seems to be broken.
    #
    #
    (test, dtc) = test_and_dtc
    obs = test.observation
    backend_ = dtc.backend
    model = mint_generic_model(backend_)
    model.set_attrs(dtc.attrs)
    pred = test.generate_prediction(model)
    print(pred)
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
        keyo = which_key(test.observation)
        dtc.observation[test.name] = test.observation[keyo]
        predo = which_key(pred)
        score = test.compute_score(test.observation,pred)
        if not hasattr(dtc,'agreement'):
            dtc.agreement = None
            dtc.agreement = {}
            #if 'mean' in test.observation.keys():
            keyo = which_key(test.observation)
            predo = which_key(pred)
            dtc.agreement[str(test)] = np.abs(test.observation[keyo]- pred[predo])
            #else:
            #    dtc.agreement[str(test)] = np.abs(test.observation['value'] - pred['value'])
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
    if 'RAW' in backend_ or 'HH' in backend_:#Backend:
        rtest = RheobaseTest(observation=place_holder,
                                name='a Rheobase test')
    else:
        rtest = RheobaseTestP(observation=place_holder,
                                name='a Rheobase test')

    dtc.rheobase = None
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
            dtc.scores[rtest.name] = 1.0 - score.norm_score
            if dtc.score is not None:
                dtc = score_proc(dtc,rtest,copy.copy(score))
            rtest.params['injected_square_current']['amplitude'] = dtc.rheobase
        else:
            dtc.rheobase = - 1.0
            dtc.scores[rtest.name] = 1.0
    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc,rtest)
    return dtc

from collections.abc import Iterable

def get_rtest(dtc):
    rtest = None
    if 'RAW' in dtc.backend or 'HH' in dtc.backend:
        if not isinstance(dtc.tests, Iterable):
            rtest = dtc.tests
        else:
            rtest = [ t for t in dtc.tests if str('RheobaseTest') == t.name ]
            print(rtest)
            rtest = rtest[0]
    else:
        if not isinstance(dtc.tests, Iterable):
            rtest = dtc.tests
        else:
            rtest = [ t for t in dtc.tests if str('RheobaseTestP') == t.name ]
            print(rtest)
            rtest = rtest[0]

    return rtest

def dtc_to_rheo(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    if not hasattr(dtc,'scores'):
        dtc.scores = None
        dtc.scores = {}
    if not hasattr(dtc,'score'):
        dtc.score = None
        dtc.score = {}
    #backend =
    model = mint_generic_model(dtc.backend)
    model.set_attrs(dtc.attrs)
    rtest = get_rtest(dtc)
    if rtest is not None:
        dtc.rheobase = rtest.generate_prediction(model)
        #print(dtc.rheobase)
        if dtc.rheobase is not None and dtc.rheobase !=-1.0:
            dtc.rheobase = dtc.rheobase['value']
            obs = rtest.observation
            score = rtest.compute_score(obs,dtc.rheobase)
            dtc.scores[rtest.name] = 1.0 - score.norm_score

            if dtc.score is not None:
                dtc = score_proc(dtc,rtest,copy.copy(score))

            rtest.params['injected_square_current']['amplitude'] = dtc.rheobase

        else:
            dtc.rheobase = - 1.0
            dtc.scores[rtest.name] = 1.0



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



def allocate_worst(tests,dtc):
    # If the model fails tests, and cannot produce model driven data
    # Allocate the worst score available.
    for t in tests:
        dtc.scores[str(t)] = 1.0
        dtc.score[str(t)] = 1.0
    return dtc

'''Should be default.
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
'''

def pred_evaluation(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    dtc = copy.copy(dtc)
    # TODO
    # phase out model path:
    # via very reduced model
    dtc.model_path = path_params['model_path']
    #preds = []
    dtc.preds = None
    dtc.preds = {}
    dtc =  dtc_to_rheo(dtc)
    dtc = format_test(dtc)
    tests = dtc.tests

    #test_and_dtc = (use_test,dtc)

    for k,t in enumerate(tests):
        if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
            t.params = dtc.vtest[k]
            test_and_models = (t, dtc)
            pred = pred_only(test_and_models)
            dtc.preds[t] = pred

        else:
            dtc.preds[t] = dtc.rheobase
    return dtc

def nunit_evaluation(dtc):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    tests = dtc.tests
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    if dtc.rheobase == -1.0 or isinstance(dtc.rheobase,type(None)):
        dtc = allocate_worst(tests, dtc)
    else:
        for k, t in enumerate(tests):
            if str('RheobaseTest') != t.name and str('RheobaseTestP') != t.name:
                t.params = dtc.vtest[k]
                score, dtc = bridge_judge((t, dtc))
                print('score', score)
                if score is not None:
                    if score.norm_score is not None:
                        dtc.scores[str(t)] = 1.0 - score.norm_score


                else:
                    print('gets to None score type')
    # compute the sum of sciunit score components.
    dtc.summed = dtc.get_ss()
    return dtc




def evaluate(dtc,regularization=True):
	# compute error using L2 regularization.
    print(dtc)
    error_length = len(dtc.scores.keys())
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
        dtcpop[0].boundary_dict = None
        dtcpop[0].boundary_dict = pop[0].boundary_dict
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
    # DO.population = None
    # DO.population = DO.grid_init()
    # DO.boundary_dict = ss
    # DO.population[0].ss = ss
    # DO.ss = ss

    # This run condition should not need same arguments as above.
    ga_out = DO.run(max_ngen = max_ngen)#offspring_size = MU, )
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
    if 'RAW' in dtcpop[0].backend  or 'HH' in dtcpop[0].backend :#Backend:
        #rtest = [ t for t in dtc.tests if str('RheobaseTest') == t.name ]
        dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
        dtcpop = list(dtcbag.map(dtc_to_rheo))
        dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
        dtcpop = list(dtcbag.map(format_test))
    else:
        dtcpop = list(map(dtc_to_rheo,dtcpop))
        dtcpop = list(map(format_test,dtcpop))
    #dtcpop = list(map(dtc_to_rheo,dtcpop))
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

    optimiser needs a stable
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
    boot_new_genes is a more standard, and less customized implementation of the code below.

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
    # LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
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
    half = int(len(a_list)/2)
    return a_list[:half], a_list[half:]



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
            aa = which_thing(a[k])
            bb = which_thing(b[k])
            t = tests[k]
            key = which_key(a[k])
            cc = which_thing(t.observation)

            grab_units = cc['standard'].units
            pp = {}
            pp[key] = np.mean([aa['standard'],bb['standard']])*grab_units
            pp['n'] = 1

            score = t.compute_score(cc,pp)
            print(score.norm_score)
            dtcb.twin = None
            dtca.twin = None
            dtca.twin = dtcb.attrs
            dtcb.twin = dtca.attrs
            # store the averaged values in the first half of genes.
            dtcb.scores[k]  = dtca.scores[k] = 1.0 - score.norm_score
            # store the averaged values in the second half of genes.
    return (dtca,dtcb)

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

    from neuronunit.optimisation.optimisations import SciUnitOptimisation
    # get waveform measurements, and store in genes.
    dtcbag = db.from_sequence(dtcpop, npartitions = NPART)

    dtcpop = list(dtcbag.map(pred_evaluation).compute())
    # divide the genes pool in half
    dtcpopa,dtcpopb = split_list(copy.copy(dtcpop))
    # average measurements between first half of gene pool, and second half.
    flat_iter = zip(dtcpopa,dtcpopb)
    #dtc_mixed = list(map(average_measurements,flat_iter))
    #print(len(dtc_mixed),len(dtcpop))
    dtcbag = db.from_sequence(flat_iter, npartitions = NPART)
    dtc_mixed = list(dtcbag.map(average_measurements))
    dtcpopa = [dtc[0] for dtc in dtc_mixed]
    dtcpopb = [dtc[1] for dtc in dtc_mixed]
    #print(len(dtcpopa))

    dtcpop = dtcpopa
    dtcpop.extend(dtcpopb)
    assert len(dtcpop) == 2*len(dtcpopb)
    return dtcpop


def boot_new_genes(number_genes,dtcpop):
    '''
    Boot strap new genes to make up for completely called onesself.
    '''
    dtcpop = copy.copy(dtcpop)
    DO = SciUnitOptimisation(offspring_size = number_genes, error_criterion = [dtcpop[0].test], boundary_dict = boundary, backend = dtcpop[0].backend, selection = str('selNSGA'))#,, boundary_dict = ss, elite_size = 2, hc=hc)
    DO.setnparams(nparams = len(free_params), boundary_dict = ss)
    DO.setup_deap()
    DO.init_pop()
    population = DO.population
    dtcpop = None
    dtcpop = update_dtc_pop(population,DO.td)
    return dtcpop

def parallel_route(pop,dtcpop,tests,td,clustered=False):
    NPART = np.min([multiprocessing.cpu_count(),len(dtcpop)])

    for d in dtcpop:
        d.tests = copy.copy(tests)
    dtcpop = list(map(format_test,dtcpop))


    if clustered == True:
        dtcpop = opt_on_pair_of_points(dtcpop)
    else:
        dtcbag = db.from_sequence(dtcpop, npartitions = NPART)
        dtcpop = list(dtcbag.map(nunit_evaluation).compute())

    #import pdb; pdb.set_trace()
    #if dtc.score is not None:
    #    dtc = score_proc(dtc,rtest,copy.copy(score))
    for i,d in enumerate(dtcpop):
        if not hasattr(pop[i],'dtc'):
            pop[i] = WSListIndividual(pop[i])
            pop[i].dtc = None
        d.get_ss()
        print(d.get_ss())
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

            if 'RAW' in dtc.backend  or 'HH' in dtc.backend :#Backend:
                #rtest = [ t for t in dtc.tests if str('RheobaseTest') == t.name ]
                dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
                dtcpop = list(dtcbag.map(dtc_to_rheo))
                dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
                dtcpop = list(dtcbag.map(format_test))
            else:
                dtcpop = list(map(dtc_to_rheo,dtcpop))
                dtcpop = list(map(format_test,dtcpop))
            dtcpop = [ dtc for dtc in dtcpop if dtc.rheobase is not None ]
            dtcpop = [ dtc for dtc in dtcpop if dtc.rheobase !=-1.0 ]
            dtcbag = db.from_sequence(dtcpop,npartitions=npartitions)
            dtcpop = list(dtcbag.map(nunit_evaluation))
            print(dtc.get_ss())
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
        pop, dtcpop = obtain_rheobase(pop, td, tests)
        pop, dtcpop = make_up_lost(pop, dtcpop, td)
        # there are many models, which have no actual rheobase current injection value.
        # filter, filters out such models,
        # gew genes, add genes to make up for missing values.
        # delta is the number of genes to replace.

    else:
        pop, dtcpop = init_pop(pop, td, tests)
    pop,dtcpop = parallel_route(pop, dtcpop, tests, td, clustered=False)
    for ind,d in zip(pop,dtcpop):
        ind.dtc = d
        if not hasattr(ind,'fitness'):
            ind.fitness = copy.copy(pop[0].fitness)
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

    #pop = copy.copy(pop)
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
