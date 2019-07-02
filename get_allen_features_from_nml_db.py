# Obtains the cell threshold, rheobase, resting v, and bias currents for
# steady state v of a cell defined in a hoc file in the given directory.
# Usage: python getCellProperties /path/To/dir/with/.hoc
#from neuronunit.tests.druckmann2013 import *


##
# This uses the docker file in this directory.
# I build it with the name efl.
# and launch it with this alias.
# alias efel='cd /home/russell/outside/neuronunit; sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/russell/outside/neuronunit:/home/jovyan/neuronunit -v /home/russell/Dropbox\ \(ASU\)/AllenDruckmanData:/home/jovyan/work/allendata efel /bin/bash'
##

from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor
try:
    import cPickle
except:
    import _pickle as cPickle
import csv
import json
import os
import re
import shutil
import string
import urllib

import inspect

import numpy as np
from matplotlib import pyplot as plt

import dask.bag as dbag # a pip installable module, usually installs without complication
import dask
import urllib.request, json
import os
import requests
from neo.core import AnalogSignal
from quantities import mV, ms, nA
from neuronunit import models
import pickle
from neuronunit.optimisation import get_neab
#from neuronunit.optimisation.optimisation_management import switch_logic#, active_values
import efel
from types import MethodType
from neuronunit.optimisation.optimisation_management import init_dm_tests
import quantities as qt
import quantities as pq
qt = pq
import pdb
#try:
#    from optimisation.optimisation_management import add_druckmann_properties_to_cells as dm
#except:
from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells as dm

from neuronunit.optimisation.optimisation_management import inject_rh_and_dont_plot
import numpy as np
import efel
from neuronunit.capabilities.spike_functions import get_spike_waveforms
import pickle
from allensdk.ephys.extract_cell_features import extract_cell_features
import pandas as pd
from allensdk.ephys.extract_cell_features import extract_cell_features
import matplotlib.pyplot as plt
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
from neuronunit.neuromldb import NeuroMLDBStaticModel
import dm_test_interoperable #import Interoperabe


def generate_prediction(self,model):
    prediction = {}
    prediction['n'] = 1
    prediction['std'] = 1.0
    prediction['mean'] = model.rheobase['mean']
    return prediction

def find_nearest(array, value):
    #value = float(value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)

def get_m_p(model,current):
    '''
    synopsis:
        get_m_p belongs to a 3 method stack (2 below)

    replace get_membrane_potential in a NU model class with a statically defined lookup table.


    '''
    return model.lookup[float(current['amplitude'])]

def update_static_model_methods(model,test_frame,lookup):
    '''
    Overwrite/ride. a NU models inject_square_current,generate_prediction methods
    with methods for querying a lookup table, such that given a current injection,
    a V_{m} is returned.
    '''
    model.lookup = lookup
    model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential
    test_frame[0][0][0].generate_prediction = MethodType(generate_prediction,test_frame[0][0][0])
    return model, test_frame

def map_to_sms(sms):
    '''
    given a list of static models, update the static models methods
    '''
    for model in sms:
        model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential
    tt[0].generate_prediction = MethodType(generate_prediction,tt[0])
    return sms

def crawl_ids(url):
    ''' move to aibs '''
    all_data = requests.get(url)
    all_data = json.loads(all_data.text)
    Model_IDs = []
    for d in all_data:
        Model_ID = str(d['Model_ID'])
        Model_IDs.append(Model_ID)
    return Model_IDs

list_to_get =[ str('https://www.neuroml-db.org/api/search?q=traub'),
    str('https://www.neuroml-db.org/api/search?q=markram'),
    str('https://www.neuroml-db.org/api/search?q=Gouwens') ]

def get_all_cortical_cells(list_to_get):
    model_ids = {}
    for url in list_to_get:
        Model_IDs = crawl_ids(url)
        parts = url.split('?q=')
        try:
            model_ids[parts[1]].append(Model_IDs)
        except:
            model_ids[parts[1]] = []
            model_ids[parts[1]].append(Model_IDs)
    with open('cortical_cells_list.p','wb') as f:
        pickle.dump(model_ids,f)

    return model_ids



def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)


def get_waveform_current_amplitude(waveform):
    return float(waveform["Waveform_Label"].replace(" nA", "")) * pq.nA


def get_static_models(cell_id):
    """
    Inputs: NML-DB cell ids, a method designed to be called inside an iteration loop.

    Synpopsis: given a NML-DB id, query nml-db, create a NMLDB static model based on wave forms
        obtained for that NML-ID.
        get mainly just waveforms, and current injection values relevant to performing druckman tests
        as well as a rheobase value for good measure.
        Update the NML-DB model objects attributes, with all the waveform data/injection values obtained for the appropriate cell IDself.
    """


    url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
    model_contents = requests.get(url)
    model_contents = json.loads(model_contents.text)
    model = NeuroMLDBStaticModel(cell_id)

    wlist = model_contents['waveform_list']
    long_squares = [ w for w in wlist if w['Protocol_ID'] == 'LONG_SQUARE' ]
    applied_current_injections = [ w for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Current" ]
    currents = [ w for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Voltage" ]
    in_current_filter = [ w for w in wlist if w["Protocol_ID"] == "SQUARE" and w["Variable_Name"] == "Voltage" ]
    rheobases = []
    for wl in long_squares:
        wid = wl['ID']
        url = str("https://neuroml-db.org/api/waveform?id=")+str(wid)
        waves = requests.get(url)
        temp = json.loads(waves.text)
        if temp['Spikes'] >= 1:
            rheobases.append(temp)
    if len(rheobases) == 0:
        return None

    in_current = []
    for check in in_current_filter:
        amp = get_waveform_current_amplitude(check)
        if amp < 0 * pq.nA:
            in_current.append(amp)
    rheobase_current = get_waveform_current_amplitude(rheobases[0])
    druckmann2013_standard_current = get_waveform_current_amplitude(currents[-2])
    druckmann2013_strong_current = get_waveform_current_amplitude(currents[-1])
    druckmann2013_input_resistance_currents = in_current
    model.waveforms = wlist
    model.protocol = {}
    model.protocol['Time_Start'] = currents[-2]['Time_Start']
    model.protocol['Time_End'] = currents[-2]['Time_End']
    model.inh_protocol = {}
    model.inh_protocol['Time_End'] = in_current_filter[0]['Time_End']
    model.inh_protocol['Time_Start'] = in_current_filter[0]['Time_Start']
    model.druckmann2013_input_resistance_currents = druckmann2013_input_resistance_currents

    model.rheobase_current = rheobase_current
    model.druckmann2013_standard_current = druckmann2013_standard_current
    model.druckmann2013_strong_current = druckmann2013_strong_current
    current = {}
    current['amplitude'] = rheobase_current
    model.vm_rheobase = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_standard_current
    model.vm15 = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_strong_current
    model.vm30 = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_input_resistance_currents[0]
    model.vminh =  model.inject_square_current(current)
    return model

def allen_format(volts,times):
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
    ext = EphysSweepSetFeatureExtractor([times],[volts])
    ext.process_spikes()

    swp = ext.sweeps()[0]
    spikes = swp.spikes()
    middle_spike_index = int(len(spikes)/2.0)
    midle_spike_info = pd.DataFrame(spikes[middle_spike_index])
    #
    #

    #allen_features = spikes
    #spike_keys = swp.spike_feature_keys()
    #swp_keys = swp.sweep_feature_keys()
    allen_features = {}
    meaned_features_overspikes = {}

    for s in swp.sweep_feature_keys():
        allen_features[s] = swp.sweep_feature(s)
        if str('isi_type') not in s:
            meaned_features_overspikes[s] = np.mean([i for i in swp.spike_feature(s) if type(i) is not type(str(''))])

    #print(swp.sweep_feature(s),s)
    #for s in swp.spike_feature_keys(): print(swp.spike_feature(s))
    #for s in swp.spike_feature_keys():

    per_spike_info = spikes
    frame = pd.DataFrame(allen_features)
    meaned_features_overspikes = pd.DataFrame(meaned_features_overspikes)
    return allen_features,frame,per_spike_info, midle_spike_info, meaned_features_overspikes

def recoverable_interuptable_batch_process():
    '''
    Synposis:

        Mass download all the NML model waveforms for all cortical regions
        And perform three types of feature extraction on resulting waveforms.

    Inputs: None
    Outputs: None in namespace, yet, lots of data written to pickle.
    '''
    all_the_NML_IDs =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))

    mid = [] # mid is a list of model identifiers.
    for v in all_the_NML_IDs.values():
        mid.extend(v[0])
    path_name = str('three_feature_folder')
    try:
        os.mkdir(path_name)
    except:
        print('directory already made :)')
    try:
        ##
        # This is the index in the list to the last NML-DB model that was analyzed
        # this index is stored to facilitate recovery from interruption
        ##
        with open('last_index.p','rb') as f:
            index = pickle.load(f)
    except:
        index = 0
    until_done = len(mid[index:-1])
    cnt = 0
    ##
    # Do the batch job, with the background assumption that some models may
    # have already been run and cached.
    ##
    while cnt <until_done-1:
        for i,mid_ in enumerate(mid[index:-1]):
            until_done = len(mid[index:-1])
            model = get_static_models(mid_)
            if type(model) is not type(None):
                three_feature_sets = three_feature_sets_on_static_models(model)
                with open(str(path_name)+str('/')+str(mid_)+'.p','wb') as f:
                    pickle.dump(three_feature_sets,f)
            with open('last_index.p','wb') as f:
                pickle.dump(i,f)
            cnt+=1




def standard_nu_tests(model,lookup,current):
	'''
	Do standard NU predictions, to do this may need to overwrite generate_prediction
	Overwrite/ride. a NU models inject_square_current,generate_prediction methods
	with methods for querying a lookup table, such that given a current injection,
	a V_{m} is returned.
	'''
	model, test_frame = update_static_model_methods(model,test_frame,lookup)
	rts,complete_map = pickle.load(open('russell_tests.p','rb'))
	local_tests = [value for value in rts['Hippocampus CA1 pyramidal cell'].values() ]
	nu_preds = []
	for t in local_tests:
	    # pred = t.generate_prediction(model)
	    try:
		pred = t.generate_prediction(model)
	    except:
		pred = None
	    nu_preds.append(pred)
	return nu_preds


def more_challenging(model):
    '''
    Isolate harder code, still wrangling data types.
    When this is done, EFEL might be able to report back about input resistance.
    '''
    single_spike = {}
    single_spike['APWaveForm'] = [ float(v) for v in model.vm_rheobase]#temp_vm
    single_spike['T'] = [ float(t) for t in model.vm_rheobase.times.rescale('ms') ]
    single_spike['V'] = [ float(v) for v in model.vm_rheobase ]#temp_vm
    single_spike['stim_start'] = [ float(model.protocol['Time_Start']) ]
    single_spike['stimulus_current'] = [ model.model.rheobase_current ]
    single_spike['stim_end'] = [ trace15['T'][-1] ]

    single_spike = [single_spike]

    trace_ephys_prop = {}
    trace_ephys_prop['stimulus_current'] = model.druckmann2013_input_resistance_currents[0]# = druckmann2013_input_resistance_currents[0]
    trace_ephys_prop['V'] = [ float(v) for v in model.vminh ]
    trace_ephys_prop['T'] = [ float(t) for t in model.vminh.times.rescale('ms') ]
    trace_ephys_prop['stim_end'] = [ trace15['T'][-1] ]
    trace_ephys_prop['stim_start'] = [ float(model.inh_protocol['Time_Start']) ]# = in_current_filter[0]['Time_End']
    trace_ephys_props = [trace_ephys_prop]

    efel_results_inh = efel.getFeatureValues(trace_ephys_props,list(efel.getFeatureNames()))#
    efel_results_ephys = efel.getFeatureValues(trace_ephys_prop,list(efel.getFeatureNames()))#


def three_feature_sets_on_static_models(model,test_frame = None):
    '''
    Conventions:
        variables ending with 15 refer to 1.5 current injection protocols.
        variables ending with 30 refer to 3.0 current injection protocols.
    Inputs:
        NML-DB models, a method designed to be called inside an iteration loop, where a list of
        models is iterated over, and on each iteration a new model is supplied to this method.

    Outputs:
        A dictionary of dataframes, for features sought according to: Druckman, EFEL, AllenSDK

    '''
    ##
    # Wrangle data to prepare for EFEL feature calculation.
    ##
    trace3 = {}
    trace3['T'] = [ float(t) for t in model.vm30.times.rescale('ms') ]
    trace3['V'] = [ float(v) for v in model.vm30]#temp_vm
    trace3['peak_voltage'] = [ np.max(model.vm30) ]

    trace3['stim_start'] = [ float(model.protocol['Time_Start']) ]
    trace3['stimulus_current'] = [ model.druckmann2013_strong_current ]
    trace3['stim_end'] = [ trace3['T'][-1] ]
    #trace0['decay_end_after_stim'] = [ trace0['T'][-1] ]# list(sm.complete['duration'])
    traces3 = [trace3]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
    trace15 = {}
    trace15['T'] = [ float(t) for t in model.vm15.times.rescale('ms') ]
    trace15['V'] = [ float(v) for v in model.vm15]#temp_vm
    trace15['peak_voltage'] = [ np.max(model.vm15) ]

    trace15['stim_start'] = [ float(model.protocol['Time_Start']) ]
    trace15['stimulus_current'] = [ model.druckmann2013_standard_current ]
    trace15['stim_end'] = [ trace15['T'][-1] ]
    #trace0['decay_end_after_stim'] = [ trace0['T'][-1] ]# list(sm.complete['duration'])
    traces15 = [trace15]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
    ##
    # Compute
    # EFEL features (HBP)
    ##

    efel_results15 = efel.getFeatureValues(traces15,list(efel.getFeatureNames()))#
    efel_results30 = efel.getFeatureValues(traces3,list(efel.getFeatureNames()))#



    df15 = pd.DataFrame(efel_results15)
    #import pdb; pdb.set_trace()
    df15['protocol'] = 1.5

    df30 = pd.DataFrame(efel_results30)
    df30['protocol'] = 3.0

    efel_frame = df15.append(df30)
    efel_frame.set_index('protocol')


    ##
    # Get Druckman features, this is mainly handled in external files.
    ##
    a = dm_test_interoperable.Interoperabe()
    a.test_setup(None,None,model= model)
    dm_test_features = a.runTest()
    dm_frame = pd.DataFrame(dm_test_features)

    ##
    # wrangle data in preperation for computing
    # Allen Features
    ##


    times = np.array([float(t) for t in model.vm30.times])
    volts = np.array([float(v) for v in model.vm30])


    ##
    # Allen Features
    ##

    allen_features,frame30, mdd30, mfos30 = allen_format(volts,times)
    frame30['protocol'] = 3.0

    ##
    # wrangle data in preperation for computing
    # Allen Features
    ##

    times = np.array([float(t) for t in model.vm15.times])
    volts = np.array([float(v) for v in model.vm15])

    ##
    # Allen Features
    ##

    allen_features,frame15, mdd15, mfos15 = allen_format(volts,times)
    frame15['protocol'] = 1.5
    allen_frame = frame30.append(frame15)
    allen_frame.set_index('protocol')

    try:
       lookup = {}
       lookup[model.druckmann2013_input_resistance_currents[0]] = model.vminh
       lookup[model.druckmann2013_standard_current] = model.vm15
       lookup[ model.druckmann2013_strong_current ] = model.vm30
       nu_preds = standard_nu_tests(model,lookup,current)

    except:
        print('standard nu tests failed.')
    #import pdb; pdb.set_trace()

    return {'efel':efel_frame,'dm':dm_frame,'allen':allen_frame,'allen_spike_data':(per_spike_info_15,per_spike_info_30)}

recoverable_interuptable_batch_process()

'''
try:
    assert 1==2
    with open('models.p','rb') as f:
        models = pickle.load(f)

except:
'''
'''
list_to_get =[ str('https://www.neuroml-db.org/api/search?q=traub'),
    str('https://www.neuroml-db.org/api/search?q=markram'),
    str('https://www.neuroml-db.org/api/search?q=Gouwens') ]


all_the_NML_IDs =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))

lll = []
for v in all_the_NML_IDs.values():
    lll.extend(v[0])

for mid in lll[0:2]:
    print(mid)
    model = get_static_models(mid)
    if type(model) is not type(None):
        three_feature_sets = three_feature_sets_on_static_models(model)
    print(three_feature_sets)
    print('gets here')
    with open('models.p','wb') as f:
        pickle.dump(models,f)






def plot_all(temps):
    for temp in temps:
        temp_vm = list(map(float, temp['Variable_Values'].split(',')))
        temp['easy_Times'] = list(map(float,temp['Times'].split(',')))

        dt = temp['easy_Times'][1]- temp['easy_Times'][0]
        temp['dt'] = dt

        temp['vm'] = AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)
        plt.plot(temp['vm'].times,temp['vm'].magnitude)#,label='ground truth')
    return temps
'''
