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
import interoperable #import Interoperabe


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
    #print('got here')
    #print(float(current['amplitude']),current)
    #print(model.lookup[float(current['amplitude'])])
    return model.lookup[float(current['amplitude'])]

def map_to_model(model,test_frame,lookup,current):
    model.lookup = lookup
    model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential
    test_frame[0][0][0].generate_prediction = MethodType(generate_prediction,test_frame[0][0][0])
    return model

def map_to_sms(sms):
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
    #value = float(value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)
    #return currents

def get_waveform_current_amplitude(waveform):
    return float(waveform["Waveform_Label"].replace(" nA", "")) * pq.nA


def get_static_models(cell_id):#,test_frame = None):
    url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
    model_contents = requests.get(url)
    model_contents = json.loads(model_contents.text)
    model = NeuroMLDBStaticModel(cell_id)
    #stability = {}
    #stability['Stability_Range_Low_Corr'] =  model_contents['model']['Stability_Range_Low_Corr']
    #stability['Max_Stable_DT_Benchmark_RunTime'] = model_contents['model']['Max_Stable_DT_Benchmark_RunTime']
    #stability['Optimal_DT_a'] = model_contents['model']['Optimal_DT_a']
    wlist = model_contents['waveform_list']
    long_squares = [ w for w in wlist if w['Protocol_ID'] == 'LONG_SQUARE' ]

    #variable_names = [  w["Variable_Name"] for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" ]
    applied_current_injections = [ w for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Current" ]

    #import pdb; pdb.set_trace()
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
    ext = EphysSweepSetFeatureExtractor([times],[volts])
    ext.process_spikes()

    swp = ext.sweeps()[0]
    spikes = swp.spikes()
    #allen_features = spikes
    #spike_keys = swp.spike_feature_keys()
    #swp_keys = swp.sweep_feature_keys()
    allen_features = {}
    for s in swp.sweep_feature_keys():
        #print(swp.sweep_feature(s),s)
        allen_features[s] = swp.sweep_feature(s)
    for s in swp.spike_feature_keys():
        #print(swp.spike_feature_keys(s),s)

        allen_features[s] = swp.spike_feature(s)
    per_spike_info = spikes
    frame = pd.DataFrame(allen_features)
    import pdb; pdb.set_trace()
    return allen_features,frame,per_spike_info

def recoverable_interuptable_batch_process():
    '''
    Mass download all the glif model parameters
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
        #pass
    try:
        with open('last_index.p','rb') as f:
            index = pickle.load(f)
    except:
        index = 0
    until_done = len(mid[index:-1])
    cnt = 0
    while cnt <until_done-1:
        for i,mid_ in enumerate(mid[index:-1]):
            until_done = len(mid[index:-1])
            print(i,mid_)
            model = get_static_models(mid_)
            if type(model) is not type(None):
                three_feature_sets = three_feature_sets_on_static_models(model)
                with open(str(path_name)+str('/')+str(mid_)+'.p','wb') as f:
                    pickle.dump(three_feature_sets,f)
            with open('last_index.p','wb') as f:
                pickle.dump(i,f)
            cnt+=1


def three_feature_sets_on_static_models(model,test_frame = None):
    ##
    # EFEL features (HBP)
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
    '''
    single_spike = {}
    single_spike['APWaveForm'] = [ float(v) for v in model.vm_rheobase]#temp_vm
    single_spike['T'] = [ float(t) for t in model.vm_rheobase.times.rescale('ms') ]
    single_spike['V'] = [ float(v) for v in model.vm_rheobase ]#temp_vm
    single_spike['stim_start'] = [ float(model.protocol['Time_Start']) ]
    single_spike['stimulus_current'] = [ model.model.rheobase_current ]
    single_spike['stim_end'] = [ trace15['T'][-1] ]

    single_spike = [single_spike]

    #model.inh_protocol['Time_Start'] #= in_current_filter[0]['Time_Start']

    '''

    #try:
        # efel_results_ephys = efel.getFeatureValues(single_spike,list(efel.getFeatureNames()))#
    #    efel_results_ephys = efel.getFeatureValues(trace_ephys_prop,list(efel.getFeatureNames()))#
    #except:
    #    print('failed on input impedance'
    #    )
    trace_ephys_prop = {}
    #print(trace3['stimulus_current'])
    #import pdb; pdb.set_trace()
    trace_ephys_prop['stimulus_current'] = model.druckmann2013_input_resistance_currents[0]# = druckmann2013_input_resistance_currents[0]
    trace_ephys_prop['V'] = [ float(v) for v in model.vminh ]
    trace_ephys_prop['T'] = [ float(t) for t in model.vminh.times.rescale('ms') ]
    trace_ephys_prop['stim_end'] = [ trace15['T'][-1] ]
    trace_ephys_prop['stim_start'] = [ float(model.inh_protocol['Time_Start']) ]# = in_current_filter[0]['Time_End']
    trace_ephys_props = [trace_ephys_prop]

    #efel_results_inh = efel.getFeatureValues(trace_ephys_props,list(efel.getFeatureNames()))#

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
    # Druckman features
    ##
    a = interoperable.Interoperabe()
    a.test_setup(None,None,model= model)
    dm_test_features = a.runTest()
    dm_frame = pd.DataFrame(dm_test_features)

    ##
    # Allen Features
    ##


    times = np.array([float(t) for t in model.vm30.times])
    volts = np.array([float(v) for v in model.vm30])
    #allen_features,frame,per_spike_info
    allen_features,frame30,per_spike_info_30 = allen_format(volts,times)
    frame30['protocol'] = 3.0

    times = np.array([float(t) for t in model.vm15.times])
    volts = np.array([float(v) for v in model.vm15])
    allen_features,frame15,per_spike_info_15 = allen_format(volts,times)
    frame15['protocol'] = 1.5
    allen_frame = frame30.append(frame15)
    allen_frame.set_index('protocol')

    import pdb; pdb.set_trace()
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
