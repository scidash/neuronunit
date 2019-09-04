from neuronunit.tests import base
from types import MethodType
from neuronunit.tests.base import VmTest
from neuronunit.tests.autils import inject_square_current, get_data_sets_from_cache, get_data_sets_from_remote
import sciunit
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
import allensdk
import pickle
import glob
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor
import pdb
import numpy as np
import copy
from neuronunit.tests.fi import RheobaseTest as generic
from sciunit.scores.complete import ZScore
import quantities as pq
from neuronunit.optimisation.optimization_management import mint_generic_model

def allen_format(volts,times,key=None,stim=None):
    '''
    Synposis:
        At its most fundamental level, AllenSDK still calls a single trace a sweep.
        In otherwords there are no single traces, but there are sweeps of size 1.
        This is a bit like wrapping unitary objects in iterable containers like [times].

    inputs:
        np.arrays of time series: Specifically a time recording vector, and a membrane potential recording.
        in floats probably with units striped away
    outputs:
        a data frame of Allen features, a very big dump of features as they pertain to each spike in a train.

        to get a managable data digest
        we out put features from the middle spike of a spike train.
    '''
    ext = EphysSweepSetFeatureExtractor([np.array(volts)],[np.array(times)])


    ext.process_spikes()
    swp = ext.sweeps()[0]
    spikes = swp.spikes()
    if len(spikes)==0:
        print('no spikes detected')
        #import pdb; pdb.set_trace()
        #return (None,None)
    meaned_features_1 = {}
    skeys = [ skey for skey in spikes[0].keys() ]
    for sk in skeys:
        if str('isi_type') not in sk:
            meaned_features_1[sk] = np.mean([ i[sk] for i in spikes if type(i) is not type(str(''))] )
    allen_features = {}
    meaned_features_overspikes = {}
    for s in swp.sweep_feature_keys():# print(swp.sweep_feature(s))
        if str('isi_type') not in s:
            allen_features[s] = swp.sweep_feature(s)
    allen_features.update(meaned_features_1)
    if key is not None:
        return allen_features[key], allen_features
    else:
        return allen_features, allen_features
try:
    data_sets = get_data_sets_from_cache(do_features=True)
    assert len(data_sets) > 1
except:
    data_sets = get_data_sets_from_remote(upper_bound=2)
    assert len(data_sets) > 1

from neuronunit.examples.hide_imports import *

#import pdb; pdb.set_trace()
def data_set(number):
    # store = [(i,len(data_sets[0][0].get_spike_times(i))) for i in sweeps ]

    specific_data = data_sets[number][0]
    numbers = specific_data.get_sweep_numbers()
    sweeps = []
    for n in numbers:
        sweeps.append(specific_data.get_sweep(n))

    stim_types = [ specific_data.get_sweep_metadata(n)['aibs_stimulus_name'] for n in numbers ]
    responses = [ specific_data.get_sweep(n)['response'] for n in numbers ]

    try:
        assert 1==2
        with open('allen_test.p','rb') as f:
            pre_obs = pickle.load(f)

    except:
        df = pd.DataFrame(rts)
        for key,v in rts.items():
            helper_tests = [value for value in v.values() ]
            break
        stim_types = [ specific_data.get_sweep_metadata(n)['aibs_stimulus_name'] for n in numbers ]
        supra_sweeps = [(i,j) for i,j in enumerate(stim_types) if str("Suprathreshold") in j ]
        analy_dict = {}
        for i in supra_sweeps:
            analy_dict[i[0]] = allensdk.ephys.extract_cell_features.extract_feature_wave_russell(responses[i[0]], stim_types[i[0]],\
            specific_data,numbers)
        #import pdb; pdb.set_trace()
        stim_amps = [ specific_data.get_sweep_metadata(n)['aibs_stimulus_amplitude_pa'] for n in numbers ]
        response_features_2 = allensdk.ephys.extract_cell_features.extract_feature_wave_russell(responses[-2], stim_types[-2],\
         specific_data,numbers)
        response_features_3 = allensdk.ephys.extract_cell_features.extract_feature_wave_russell(responses[-3], stim_types[-3],\
         specific_data,numbers)

        responses = [ specific_data.get_sweep(n)['response'] for n in numbers ]
        one_sweep = [ stim_amps[-2],response_features_2[list(response_features_2.keys())[-2]],helper_tests, responses[-2] ]
        two_sweep = [ stim_amps[-3],response_features_3[list(response_features_3.keys())[-3]],helper_tests, responses[-3] ]
        #import pdb
        #pdb.set_trace()

        pre_obs = [specific_data.file_name,one_sweep,two_sweep]

        observation = {}
        observation['value'] = pre_obs
        with open('allen_test{0}.p'.format(number),'wb') as f:
            pickle.dump(pre_obs,f)
        return pre_obs
pre_obs = data_set(0)
#['isi_cv'], 'spikes', 'mean_isi', 'id', 'adapt', 'latency', 'median_isi', 'avg_rate', 'first_isi'])
#import pdb
#pdb.set_trace()
'''
data_keys  = list(response_features.keys())
cv = {}
cv['mean'] = response_features[data_keys[0]]['isi_cv']
latency = {}
latency['mean'] = response_features[data_keys[0]]['latency']
avg_rate = {}
avg_rate['mean'] = pre_obs[data_keys[0]]['avg_rate']
median_isi = {}
median_isi['mean'] = response_features[data_keys[0]]['median_isi']
upstroke = {}
upstroke['mean'] = response_features[data_keys[0]]['spikes'][0]['upstroke_v']
upstroke['std'] = 1

width = {}
width['mean'] = response_features[data_keys[0]]['spikes'][0]['width']
height = {}
height['mean'] = response_features[data_keys[0]]['spikes'][0]['peak_v']
###
#
cv_test = sciunit.Test(cv)
cv_test.name = 'isi_cv'
latency_test = sciunit.Test(latency)
latency_test.name = 'latency'
avg_rate_test = sciunit.Test(avg_rate)
avg_rate_test.name = 'avg_rate'
median_isi_test = sciunit.Test(median_isi)
median_isi_test.name = 'median_isi'
width_test = sciunit.Test(width)
width_test.name = 'width'
upstroke_test = sciunit.Test(upstroke)
upstroke_test.name = 'upstroke_v'
peak_test = sciunit.Test(height)
peak_test.name = 'peak_v'

test_collection = [cv_test,latency_test,avg_rate_test,median_isi_test,width_test,upstroke_test,peak_test]

def dtc_to_model(dtc):
    # If  test taking data, and objects are present (observations etc).
    # Take the rheobase test and store it in the data transport container.
    if not hasattr(dtc,'scores'):
        dtc.scores = None
    if type(dtc.scores) is type(None):
        dtc.scores = {}
    model = mint_generic_model(dtc.backend)
    print(dtc.backend)
    model.attrs = dtc.attrs

    return model

def generate_prediction(self, model):
    key = copy.copy(self.name)
    if model.static == True:
        model.inject_square_current(model.druckmann2013_strong_current)
    else:
        if hasattr(model.dtc,'rheobase'):
            rheobase = model.dtc.rheobase
        keyed = {}
        keyed['injected_square_current'] = {}
        DURATION = 1000.0*pq.ms
        DELAY = 100.0*pq.ms
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*1.5*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = 1.5*rheobase
        keyed['injected_square_current']['delay']= DELAY
        keyed['injected_square_current']['duration'] = DURATION
        #print('gets to a')
        model = dtc_to_model(model.dtc)

        model.vm15 = None
        model.inject_square_current(keyed['injected_square_current'])
        #model.finalize()
        model.vm15 = model.get_membrane_potential()
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*3.0*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = 3.0*rheobase
        keyed['injected_square_current']['delay']= DELAY
        keyed['injected_square_current']['duration'] = DURATION
        #model = dtc_to_model(model.dtc)
        model.vm30 = None
        model.inject_square_current(keyed['injected_square_current'])
        #model.finalize()
        model.vm30 = model.get_membrane_potential()
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase
        keyed['injected_square_current']['delay']= DELAY
        keyed['injected_square_current']['duration'] = DURATION
        #model = dtc_to_model(model.dtc)
        #vmrh = model.inject_square_current(keyed['injected_square_current'])
        model.vmrh = None
        model.inject_square_current(keyed['injected_square_current'])
        model.finalize()
        model.vmrh = model.get_membrane_potential()
        stim = keyed['injected_square_current']
    prediction = {}

    if not hasattr(model,'lookup'):# is None:
        #import pdb; pdb.set_trace()
        volts = np.array([float(v) for v in model.vm15.magnitude])
        times = np.array([float(t) for t in model.vm15.times])
        #print(volts,'gets to c')
        pred,model.lookup = allen_format(volts,times,key,stim)
    if model.lookup is None:
        volts = np.array([float(v) for v in model.vm15.magnitude])
        times = np.array([float(t) for t in model.vm15.times])

        pred,model.lookup = allen_format(volts,times,key,stim)

        print(pred,model.lookup)
        import pdb; pdb.set_trace()
    else:
        pred = model.lookup[key]
    prediction['model'] = model
    prediction['value'] = pred
    prediction['mean'] = pred
    return prediction

def judge(model,test,prediction):

    score = VmTest.compute_score(test,test.observation,prediction)
    return score, model

#cv_test.compute_score = MethodType(compute_score,cv_test)
def mutate_test(test,generic):
    generic = generic()
    generic.key = None
    generic.key = test.name
    generic.observation = test.observation
    generic.name = test.name
    #generic.generate_prediction = MethodType(generate_prediction,generic)
    generic.judge = MethodType(judge,generic)
    return generic

generics = [ copy.copy(generic) for t in test_collection ]


for i,(t,g) in enumerate(zip(test_collection,generics)): test_collection[i] = mutate_test(t,g)
names = [ t.name for t in test_collection ]
keys = [ t.key for t in test_collection ]

test_collection[-2].score_type = ZScore
#scores = [ judge(model,t) for t in test_collection ]
'''
