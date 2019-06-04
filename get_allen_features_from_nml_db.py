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
from neuronunit.optimisation.optimisation_management import switch_logic#, active_values
import efel
from types import MethodType
from neuronunit.optimisation.optimisation_management import init_dm_tests

import pdb
try:
    from optimisation.optimisation_management import add_druckmann_properties_to_cells as dm
except:
    from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells as dm

from optimisation.optimisation_management import inject_rh_and_dont_plot
import numpy as np
import efel
from capabilities.spike_functions import get_spike_waveforms
import pickle
from allensdk.ephys.extract_cell_features import extract_cell_features

def crawl_ids(url):
    ''' move to aibs '''
    all_data = requests.get(url)
    all_data = json.loads(all_data.text)
    print(all_data)


    Model_IDs = []
    for d in all_data:
        Model_ID = str(d['Model_ID'])
        try:
            url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+Model_ID+str('&version=NEURON')
            print(url)
            urllib.request.urlretrieve(url,Model_ID)
            Model_IDs.append(Model_ID)
        except:
            url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+Model_ID+str('&version=NeuroML')
            print(url)
            urllib.request.urlretrieve(url,Model_ID)
            os.system(str('unzip ')+str(d['Model_ID'])+('*'))
            os.chdir(('*')+str(d['Model_ID'])+('*'))
            try:
                os.system(str('nrnivmodl *.mod'))
            except:
                print('No MOD files')
            try:
                os.system(str('pynml hhneuron.cell.nml -neuron'))
            except:
                print('No NeuroML files')

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
        authors[parts[1]] = Model_IDs


def get_wave_forms(cell_id):
    url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
    waveids = requests.get(url)
    waveids = json.loads(waveids.text)
    wlist = waveids['waveform_list']
    waves_to_get = []
    for wl in wlist:
        waves_to_test = {}
        wid = wl['ID']
        url = str("https://neuroml-db.org/api/waveform?id=")+str(wid)
        waves = requests.get(url)
        temp = json.loads(waves.text)
        if 'NOISE' in temp['Protocol_ID']:

            pass
        if 'RAMP' in temp['Protocol_ID']:
            pass
        if 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID']:
            print((temp['Waveform_Label']))
            pass
        if type(temp['Waveform_Label']) is type(None):
            break
        if str("1.0xRB") in temp['Waveform_Label'] or str("1.25xRB") in temp['Waveform_Label']:
            print(temp['Protocol_ID'])
            temp_vm = list(map(float, temp['Variable_Values'].split(',')))
            waves_to_test['Times'] = list(map(float,temp['Times'].split(',')))
            waves_to_test['DURATION'] = temp['Time_End'] -temp['Time_Start']
            waves_to_test['DELAY'] = temp['Time_Start']
            waves_to_test['Time_End'] = temp['Time_End']
            waves_to_test['Time_Start'] = temp['Time_Start']
            dt = waves_to_test['Times'][1]- waves_to_test['Times'][0]
            waves_to_test['vm'] = AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)


            sm = models.StaticModel(waves_to_test['vm'])
            sm.complete = None
            sm.complete = temp
            sm.complete['vm'] = waves_to_test['vm']


            trace0 = {}
            DURATION = sm.complete['Time_End'] -sm.complete['Time_Start']

            trace0['T'] = waves_to_test['Times']

            trace0['V'] = temp_vm
            trace0['stim_start'] = [sm.complete['Time_Start']]#rtest.run_params[]
            trace0['stim_end'] = [sm.complete['Time_End'] ]# list(sm.complete['duration'])
            traces0 = [trace0]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
            print(temp['Spikes'])
            if temp['Spikes'] and np.min(temp_vm)<0:

                traces_results = efel.getFeatureValues(traces0,list(efel.getFeatureNames()))#
                for v in traces_results:
                    for key,value in v.items():
                        if type(value) is not type(None):
                            #pass
                            print(key,value)


                [(model,times,vm)] = pickle.load(open('efel_practice.p','rb'))

                waveforms = get_spike_waveforms(vm)

                trace1 = {}
                trace1['T'] = waveforms[:,0].times
                trace1['V'] = waveforms[:,0]

                trace1['T'] = [ float(t) for t in trace1['T'] ]
                trace1['V'] = [ float(v) for v in trace1['V'] ]
                trace1['stim_start'] = [ 0 ] #[sm.complete['Time_Start']]#rtest.run_params[]
                trace1['stim_end'] = [ 0 + float(np.max(trace1['T'])) ]# list(sm.complete['duration'])
                traces1 = [trace1]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
                import pdb; pdb.set_trace()

                traces_results = efel.getFeatureValues(traces1,list(efel.getFeatureNames()))#
                print(trace_results)
                dm_properties['efel'].append(traces_results)
                dtc.efel_properties = None
                dtc.efel_properties = dm_properties['efel']
            else:
                print(temp['Spikes'],np.min(temp_vm))
                pass




        if 'SHORT_SQUARE' in temp['Protocol_ID'] and not 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID'] or 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID']:
            try:
                parts = temp['Waveform_Label'].split(' ')
                print(' value ',parts[0], ' units ',parts[1])
                waves_to_test['prediction'] = float(parts[0])*nA# '1.1133 nA',

                temp_vm = list(map(float, temp['Variable_Values'].split(',')))

                waves_to_test['Times'] = list(map(float,temp['Times'].split(',')))
                waves_to_test['DURATION'] = temp['Time_End'] -temp['Time_Start']
                waves_to_test['DELAY'] = temp['Time_Start']
                waves_to_test['Time_End'] = temp['Time_End']
                waves_to_test['Time_Start'] = temp['Time_Start']
                dt = waves_to_test['Times'][1]- waves_to_test['Times'][0]
                waves_to_test['vm'] = AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)
                waves_to_test['everything'] = temp
                waves_to_get.append(waves_to_test)

            except:

                if temp['Waveform_Label'] is None:
                    pass
                pass
    return waves_to_get

try:
    with open('static_models.p','rb') as f: sms = pickle.load(f)
    with open('waves.p','rb') as f: wlist = pickle.load(f)
except:
    waves = get_wave_forms(str('NMLCL001129'))
    with open('waves.p','wb') as f: pickle.dump(waves,f)


protocols = [ w['everything']['Protocol_ID'] for w in waves ]
ss = [ w['everything'] for w in waves if w['everything']['Protocol_ID'] == 'SHORT_SQUARE' ]

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

electro_path = str(os.getcwd())+'/examples/pipe_tests.p'

assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    test_frame = pickle.load(f)

def generate_prediction(self,model):
    prediction = {}
    prediction['n'] = 1
    prediction['std'] = 1.0
    prediction['mean'] = model.rheobase['mean']
    return prediction

test_scores = []
tt = [tests for tests in test_frame[0][0] ]
for t in tt: t.generate_prediction = MethodType(generate_prediction,t)
params = {}


def get_m_p(cls,params = {}):
    return model.get_membrane_potential()
for model in sms:
    model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential

tt[0].generate_prediction = MethodType(generate_prediction,tt[0])

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


flat_iter = [(t,sm) for t in tt for sm in sms]
import pdb
waves = get_wave_forms(str('NMLCL001129'))

for t,sm in flat_iter:
    sm._backend = None
    if t.active:
        t.params = {}
        t.params['injected_square_current'] = None

        #score = rtest.judge(model)
        results = sm.get_membrane_potential()
        value = float(np.max(current_injections[0]['vm']))
        dm_tests = init_dm_tests(value,1.5*value)
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
