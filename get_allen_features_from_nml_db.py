# Obtains the cell threshold, rheobase, resting v, and bias currents for
# steady state v of a cell defined in a hoc file in the given directory.
# Usage: python getCellProperties /path/To/dir/with/.hoc
#from neuronunit.tests.druckmann2013 import *
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
    print(float(current['amplitude']),current)
    print(model.lookup[float(current['amplitude'])])
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
    authors = {}
    for url in list_to_get:
        Model_IDs = crawl_ids(url)
        parts = url.split('?q=')
        try:
            authors[parts[1]].append(Model_IDs)
        except:
            authors[parts[1]] = []
            authors[parts[1]].append(Model_IDs)
    with open('cortical_cells_list.p','wb') as f:
        pickle.dump(authors,f)

    return authors

authors = get_all_cortical_cells(list_to_get)

def plot_all(temps):
    for temp in temps:
        temp_vm = list(map(float, temp['Variable_Values'].split(',')))
        temp['easy_Times'] = list(map(float,temp['Times'].split(',')))

        dt = temp['easy_Times'][1]- temp['easy_Times'][0]

        temp['vm'] = AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)
        plt.plot(temp['vm'].times,temp['vm'].magnitude)#,label='ground truth')
    #plt.show()
    return temps

def find_nearest(array, value):
    #value = float(value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)

def get_wave_forms(cell_id):#,test_frame = None):


    url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
    model_contents = requests.get(url)
    model_contents = json.loads(model_contents.text)
    stability = {}
    stability['Stability_Range_Low_Corr'] =  model_contents['model']['Stability_Range_Low_Corr']
    stability['Max_Stable_DT_Benchmark_RunTime'] = model_contents['model']['Max_Stable_DT_Benchmark_RunTime']
    stability['Optimal_DT_a'] = model_contents['model']['Optimal_DT_a']
    wlist = model_contents['waveform_list']
    long_squares = [ w for w in wlist if w['Protocol_ID'] == 'LONG_SQUARE' ]

    waves_to_get = []
    temps = []
    for wl in long_squares:
        wid = wl['ID']
        url = str("https://neuroml-db.org/api/waveform?id=")+str(wid)
        waves = requests.get(url)
        temp = json.loads(waves.text)
        if temp['Spikes'] > 0:
            temps.append(temp)
    with open('temps.p','wb') as f:
        pickle.dump(temps,f)
    return temps


def take_temps(temps,test_frame = None):
    if test_frame == None:
        electro_path = str(os.getcwd())+str('/neuronunit/examples/pipe_tests.p')
        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f: test_frame = pickle.load(f)

    temps = plot_all(temps)
    injections = [ float(t['Waveform_Label'].split(' ')[0]) for t in temps ]

    (rheo15, idx15) = find_nearest(injections, float(injections[0])*1.5)
    (rheo30, idx30) = find_nearest(injections, float(injections[0])*3.0)
    new_temps = [ temps[idx15], temps[idx30] ]
    lookup= {}
    for i in [idx15,idx30]:
        current = injections[i]
        response = temps[i]['vm']
        lookup[current] = response
    #params = {}

    for i,temp in enumerate(new_temps):
        waves_to_test = {}

        waves_to_test['easy_Times'] = temp['easy_Times'] # list(map(float,temp['Times'].split(',')))
        waves_to_test['DURATION'] = temp['Time_End'] -temp['Time_Start']
        waves_to_test['DELAY'] = temp['Time_Start']
        waves_to_test['Time_End'] = temp['Time_End']
        waves_to_test['Time_Start'] = temp['Time_Start']

        ## Make a static NEURONUNIT MODEL
        trace0 = {}
        sm = models.StaticModel(temp['vm'])
        sm.complete = temp
        sm.complete['vm'] = temp['vm']
        sm = map_to_model(sm,test_frame,lookup,current)
        #plt.plot(sm.get_membrane_potential().times,sm.get_membrane_potential())#,label='ground truth')
        DURATION = sm.complete['Time_End'] -sm.complete['Time_Start']
        trace0['T'] = waves_to_test['easy_Times']
        trace0['V'] = temp['vm']#temp_vm
        trace0['stim_start'] = [sm.complete['Time_Start']]#rtest.run_params[]
        trace0['stim_end'] = [sm.complete['Time_End'] ]# list(sm.complete['duration'])
        #params['delay'] =  sm.complete['Time_Start']
        #params['duration'] = sm.complete['Time_End'] - sm.complete['Time_Start']
        #params['amplitude'] = injections[i] * qt.pA
#
        params = {
            'injected_square_current': {
                'delay': sm.complete['Time_Start'] * pq.ms,
                'duration': sm.complete['Time_End']* pq.ms - sm.complete['Time_Start'] * pq.ms,
                'amplitude':  injections[i] * qt.pA
            },
            'threshold': -20 * pq.mV,
            'beginning_threshold': 12.0 * pq.mV/pq.ms,
            'ap_window': 10 * pq.ms,
            'repetitions': 1,
        }
        traces0 = [trace0]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
        #print(temp['Spikes'])



    #if temp['Spikes'] and np.min(temp_vm)<0:
    traces_results = efel.getFeatureValues(traces0,list(efel.getFeatureNames()))#
    for v in traces_results:
        for key,value in v.items():
            if type(value) is not type(None):
                print(key,value)
                pass

    in_resistance = traces_results[0]['ohmic_input_resistance']
    sag = traces_results[0]['sag_amplitude']
    cv = traces_results[0]['ISI_CV']
    isis = traces_results[0]['ISIs']
    median_isis = np.median(isis)

    #results = sm.get_membrane_potential()


    rheobase = float(temp['Waveform_Label'].split(' ')[0])

    #import pdb
    #pdb.set_trace()
    dm_tests = init_dm_tests(params,rheo15*qt.pA,rheo30*qt.pA)
    #for d in dm_tests: if str(neuronunit.tests.druckman2013.ISIMedianTest) in d: pdb.set_trace()
    isi_median = [d for d in dm_tests if str("ISIMedianTest")==str(d) ]
    isi_median.generate_prediction(sm)
    isi_median.generate_repetition_prediction()
    delay = [ d for d in dm_tests if str("AP1DelayMeanTest")== str(d) ]
    accom = [ d for d in dm_tests if str("InitialAccommodationMeanTest")== str(d)]
    #dm_tests = init_dm_tests(rheo15,rheo30)

    predictions = []
    print(params)
    import pdb;
    pdb.set_trace()
    #[
    for dm in dm_tests:
        dm.params = params
    [ dm.get_APs(sm) for dm in dm_tests ]

    for dm in dm_tests:
        try:
            predictions.append(dm.generate_prediction(sm))
        except:

            pass
    print(predictions)
    import pdb;
    pdb.set_trace()

    print(predictions)
    efel_to_data_frame = []
    for i,k in enumerate(traces_results[0].keys()):
        efel_to_data_frame.append({k:v})
    pd.DataFrame(efel_to_data_frame)
    return waves_to_get


try:
    with open('temps.p','rb') as f:
        temps = pickle.load(f)
except:
    temps = get_wave_forms(str('NMLCL001129'))

take_temps(temps)
    #with open('waves.p','wb') as f:
    #    pickle.dump(waves,f)

import pdb
pdb.set_trace()

protocols = [ w['everything']['Protocol_ID'] for w in waves ]
ss = [ w['everything'] for w in waves if w['everything']['Protocol_ID'] == 'LONG_SQUARE' ]

not_current_injections = [ w for w in waves if w['everything']['Variable_Name'] != str('Current') ]
sms = []
for w in not_current_injections:
    sm = models.StaticModel(w['vm'])
    sm.rheobase = {}
    sm.rheobase['mean'] = w['prediction']
    sm.complete = w
    sms.append(sm)
with open('static_models.p','wb') as f:
    pickle.dump(sms,f)
current_injections = [ w for w in waves if w['everything']['Variable_Name'] == str('Current') ]


import pdb; pdb.set_trace()
def active_values(keyed,rheobase,square = None):
    keyed['injected_square_current'] = {}
    if square == None:
        DURATION = 1000.0*pq.ms
        DELAY = 100.0*pq.ms
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase
    else:
        DURATION = square['Time_End'] -square['Time_Start']
        DELAY = square['Time_Start']
        keyed['injected_square_current']['amplitude'] = square['prediction']#value'])*pq.pA

    keyed['injected_square_current']['delay']= DELAY
    keyed['injected_square_current']['duration'] = DURATION

    return keyed



tt = switch_logic(tt) # for tests in tt ]
for t in tt:
    for sm in sms:
        rheobase = sm.complete['prediction']
        t.params = {}
        t.params = active_values(t.params,rheobase,square=sm.complete)

#def inject_square_current():#
#    return


flat_iter = [(t,sm) for t in tt for sm in sms]
import pdb
waves = get_wave_forms(str('NMLCL001129'))

for t,sm in flat_iter:
    sm._backend = None
    if t.active:
        t.params = {}
        t.params['injected_square_current'] = None
        results = sm.get_membrane_potential()
        value = float(np.max(current_injections[0]['vm']))
        dm_tests = init_dm_tests(1.5*value,3.0*value)
        predictions = [ dm.generate_prediction(sm) for dm in dm_tests ]
        trace = {}
        trace['T'] = sm.complete['Times']
        trace['V'] = results
        trace['stim_start'] = [sm.complete['Time_Start']]
        DURATION = sm.complete['Time_End'] -sm.complete['Time_Start']
        trace['stim_end'] = [sm.complete['Time_End'] ]
        traces = [trace]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
        traces_results = efel.getFeatureValues(traces,list(efel.getFeatureNames()))#


        for v in traces_results:
            for key,value in v.items():
                if type(value) is not type(None):
                    print(key,value)
        try:
            test_scores.append(t.judge(sm))
            print(test_scores[-1],t.name)
        except:
            test_scores.append(t.judge(sm))
            print(test_scores[-1],t.name)
            print('active skipped: ',t.name)
